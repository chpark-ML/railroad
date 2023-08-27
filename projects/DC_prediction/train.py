import importlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Type, TypeVar, Union

import hydra
import numpy as np
import omegaconf
import optuna
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.nn.parallel as tnp
import tqdm
from sklearn import metrics
from torch.utils.data.distributed import DistributedSampler

import projects.DC_prediction.utils.constants as C
from projects.DC_prediction.utils import experiment_tool as et
from projects.DC_prediction.utils.enums import RunMode
from projects.DC_prediction.utils.utils import (
    get_binary_classification_metrics, _seed_everything, print_config, set_config, get_torch_device_string)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Trainer')


@dataclass
class Metrics():
    loss: float = np.inf

    def __str__(self):
        return f'loss_{self.loss:.4f}'

    def get_representative_metric(self):
        """
        Returns: float type evaluation metric
        """
        return self.loss


def _check_any_nan(arr):
    if torch.any(torch.isnan(arr)):
        import pdb
        pdb.set_trace()


def _get_loaders_and_trainer(config, optuna_trial=None):
    # Pretty print config using Rich library
    if config.get('print_config'):
        print_config(config, resolve=True)

    # Set seed for random number generators in pytorch, numpy and python.random
    if 'seed' in config:
        _seed_everything(config.seed)

    # Data Loaders
    logger.info(f"Instantiating dataloader <{config.loader._target_}>")
    run_modes = [RunMode(m) for m in config.run_modes] if 'run_modes' in config else [x for x in RunMode]
    loaders = {mode: hydra.utils.instantiate(config.loader,
                                                dataset={'mode': mode}, shuffle=(mode == RunMode.TRAIN),
                                                drop_last=(mode == RunMode.TRAIN)) for mode in run_modes}

    # Trainers
    module_name, _, trainer_class = config.trainer._target_.rpartition('.')
    module = importlib.import_module(module_name)
    class_ = getattr(module, trainer_class)

    model = hydra.utils.instantiate(config.model)
    logging_tool = et.load_logging_tool(config=config)
    trainer = class_.hydrate_trainer(config, loaders, model, logging_tool, optuna_trial=optuna_trial)

    return loaders, trainer, logging_tool


def train(config: omegaconf.DictConfig, optuna_trial=None) -> object:
    # Convenient config setup: easier access to debug mode, etc.
    config: omegaconf.DictConfig = set_config(config)

    loaders, trainer, logging_tool = _get_loaders_and_trainer(config, optuna_trial)

    best_model_metrics, best_model_test_metrics = None, None
    try:
        best_model_metrics = trainer.fit(loaders)
        if config.eval_best_ckpt:
            trainer.test(loaders)
        logging_tool.log_model(trainer.model, "model")

    except KeyboardInterrupt:
        # Save intermediate output on keyboard interrupt
        model_path = 'keyboard-interrupted-final.pth'
        trainer.save_checkpoint(model_path)
        logging_tool.raise_keyboardinterrupt()

    except optuna.TrialPruned:
        if logging_tool:
            logging_tool.end_run()
        raise optuna.TrialPruned()

    except Exception as e:
        logger.error(f"Error while training: {e}")
        logger.exception(e)
        if logging_tool:
            logging_tool.raise_error(error=e)

    if logging_tool:
        logging_tool.end_run()

    optimizing_metric = trainer.optimizing_metric(best_model_metrics) if best_model_metrics else None

    logger.info(f"Optimizing metrics is {optimizing_metric}")

    return optimizing_metric


class Trainer():
    """Trainer to train model"""

    def __init__(self, model, optimizer, scheduler, criterion, logging_tool, gpus, fast_dev_run, max_epoch,
                 log_every_n_steps=1, test_epoch_start=0, resume_from_checkpoint=False, benchmark=False,
                 deterministic=True, fine_tune_info=None, early_stop_patience: int = None, 
                 use_amp: bool = True, optuna_trial: optuna.Trial = None, **kwargs) -> None:
        self.path_best_model = ''
        self.epoch_best_model = 0
        self.optimizer = optimizer
        self.scheduler = SchedulerTool(scheduler)
        self.criterion = criterion
        self.logging_tool = logging_tool
        self.use_amp = use_amp
        self.scaler = amp.GradScaler()

        self.epoch = 0
        self.resume_epoch = 0

        # Training configurations
        self.max_epoch = max_epoch
        self.fast_dev_run = fast_dev_run
        self.log_every_n_steps = log_every_n_steps
        self.test_epoch_start = test_epoch_start
        self.resume_from_checkpoint = resume_from_checkpoint
        # This constant is used in fit function for counting epochs for early stop functionality.
        self.early_stop_patience = early_stop_patience if early_stop_patience else max_epoch
        self.optuna_trial = optuna_trial

        torch_device = get_torch_device_string(gpus)
        if torch_device.startswith('cuda'):
            cudnn.benchmark = benchmark
            cudnn.deterministic = deterministic
        logger.info(f"Using torch device: {torch_device}")
        self.device = torch.device(torch_device)
        self.model = model.to(self.device)

    @classmethod
    def hydrate_trainer(cls, config: omegaconf.DictConfig, loaders, model, logging_tool, optuna_trial=None) -> T:
        # Init model
        logger.info(f'Instantiating model <{config.model._target_}>')

        # Init optimizer
        optimizer = hydra.utils.instantiate(config.optim, model.parameters())
        if "steps_per_epoch" in config.scheduler:
            config.scheduler["steps_per_epoch"] = len(loaders[RunMode.TRAIN])
        scheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
        criterion = hydra.utils.instantiate(config.criterion)

        # Init trainer
        logger.info(f'Instantiating trainer <{config.trainer._target_}>')
        return hydra.utils.instantiate(config.trainer,
                                       model=model,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       criterion=criterion,
                                       logging_tool=logging_tool,
                                       use_amp=config.trainer.use_amp,
                                       optuna_trial=optuna_trial)

    def train_epoch(self, epoch, train_loader) -> Metrics:
        self.model.train()
        train_losses = []
        start = time.time()
        for i, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            x = data['x'].to(self.device)
            y = data['y'].to(self.device)
            yaw = data['yaw'].to(self.device)
            _check_any_nan(x)
            _check_any_nan(y)

            # forward propagation
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x, yaw)
                loss = self.criterion(logits, y)["hybrid"]
                train_losses.append(loss.detach())

            # set trace for checking nan values
            if torch.any(torch.isnan(loss)):
                import pdb
                pdb.set_trace()
                is_param_nan = torch.stack([torch.isnan(p).any() for p in self.model.parameters()]).any()
                continue

            # Backpropagation
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

            if i % self.log_every_n_steps == 0:
                idx = i + 1
                global_step = epoch * len(train_loader) + idx
                batch_time = time.time() - start
                self.log_metrics(RunMode.TRAIN.value, global_step, Metrics(loss.detach()),
                                 log_prefix=f'[{epoch}/{self.max_epoch}] [{i}/{len(train_loader)}]',
                                 mlflow_log_prefix='STEP',
                                 duration=batch_time)
                start = time.time()

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                break
            self.scheduler.step("step")
        self.scheduler.step("epoch")

        train_loss = torch.stack(train_losses).sum().item()
        return Metrics(train_loss / len(train_loader))

    def optimizing_metric(self, metrics: Metrics):
        return metrics.loss

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_initial_model_metric(self):
        return Metrics()

    def get_metrics(self, preds, annots):
        losses = self.criterion(preds, annots)["MAPE"]
        result_dict = {}

        return losses.detach(), result_dict

    def _inference(self, loader):
        list_logits = []
        list_annots = []

        for data in tqdm.tqdm(loader):
            x = data['x'].to(self.device)
            y = data['y'].to(self.device)
            yaw = data['yaw'].to(self.device)
            _check_any_nan(x)
            
            if loader.dataset.test_input_length == loader.dataset.window_length:
                logits = self.model(x, yaw)
            else:
                _interval = loader.dataset.window_length // 2
                _num_windowing = 1 + int(np.ceil((x.size(3) - loader.dataset.window_length) / _interval))
                _expected_size = loader.dataset.window_length + int(np.ceil((x.size(3) - loader.dataset.window_length) / _interval)) * _interval
                diff_t = _expected_size - loader.dataset.test_input_length
                x_padded = F.pad(x, [0, diff_t, 0, 0])
                logits = torch.zeros((x.size(0), 1, len(C.PREDICT_COLS), x.size(3))).to(self.device)
                counters = torch.zeros((x.size(0), 1, len(C.PREDICT_COLS), x.size(3))).to(self.device)
                for i in range(_num_windowing):
                    start_index = i * _interval
                    end_index = start_index + loader.dataset.window_length 
                    logits_padded = self.model(x_padded[:, :, :, start_index: end_index], yaw)
                    logits[:, :, :, start_index: end_index] = logits_padded
                    counters[:, :, :, start_index: end_index] += 1
                logits = (logits / counters)[:, :, :, :x.size(3)]

            list_logits.append(logits)
            list_annots.append(y)

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                # FIXME: progress bar does not update when 'fast_dev_run==True'
                break

        preds = torch.vstack(list_logits)
        annots = torch.vstack(list_annots)

        return preds, annots

    @torch.no_grad()
    def validate_epoch(self, epoch, val_loader) -> Metrics:
        self.model.eval()
        preds, annots = self._inference(val_loader)
        loss, dict_metrics = self.get_metrics(preds, annots)
        self.scheduler.step("epoch_val", loss)

        return Metrics(loss)

    @torch.no_grad()
    def test_epoch(self, epoch, test_loader) -> Metrics:
        self.model.eval()
        preds, annots = self._inference(test_loader)
        loss, dict_metrics = self.get_metrics(preds, annots)

        return Metrics(loss)

    def save_best_metrics(self, val_metrics: Metrics, best_metrics: Metrics, epoch) -> (
            object, bool):
        found_better = False
        if val_metrics is None:
            found_better = True
            model_path = f'model.pth'
            self.path_best_model = model_path
            self.epoch_best_model = epoch
            self.save_checkpoint(model_path)

        elif val_metrics.loss < best_metrics.loss:
            found_better = True
            model_path = f'model.pth'
            logger.info(f"loss improved from {best_metrics.loss:4f} to {val_metrics.loss:4f}, "
                        f"saving model to {model_path}.")
            best_metrics = val_metrics
            self.path_best_model = model_path
            self.epoch_best_model = epoch
            self.save_checkpoint(model_path)

        return best_metrics, found_better

    def save_checkpoint(self, model_path):
        checkpoint = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.use_amp:
            checkpoint['scaler'] = self.scaler.state_dict()
        torch.save(checkpoint, model_path)

    def load_checkpoint(self, model_path):
        """Loads checkpoint from directory"""
        assert os.path.exists(model_path)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        logger.info(f'Model loaded from {model_path}')

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
            
    def _run_epoch(self, epoch, train_loader, val_loader, test_loader, best_model_metrics: object):
        self.epoch = epoch

        # Train for one epoch
        start = time.time()
        train_metrics = self.train_epoch(epoch, train_loader)
        self.log_lr(self.get_lr(), epoch, log_prefix=f'[{epoch}/{self.max_epoch}]', mlflow_log_prefix='EPOCH')
        self.log_metrics(RunMode.TRAIN.value, epoch, train_metrics, log_prefix=f'[{epoch}/{self.max_epoch}]',
                         mlflow_log_prefix='EPOCH', duration=time.time() - start)

        # Validation for one epoch
        val_metrics = None
        if len(val_loader) != 0:
            val_metrics = self.validate_epoch(epoch, val_loader)
            self.log_metrics(RunMode.VALIDATE.value, epoch, val_metrics, mlflow_log_prefix='EPOCH')

        if self.optuna_trial is not None and val_metrics is not None:
            # Report intermediate objective value.
            self.optuna_trial.report(val_metrics.get_representative_metric(), epoch)

            # Handle pruning based on the intermediate value.
            if self.optuna_trial.should_prune():
                raise optuna.TrialPruned()

        # Test if possible
        test_metrics = None
        if test_loader and epoch >= self.test_epoch_start:
            test_metrics = self.test_epoch(epoch, test_loader)
            self.log_metrics(RunMode.TEST.value, epoch, test_metrics, mlflow_log_prefix='EPOCH')

        # Save model and return metrics
        best_metrics, found_better = self.save_best_metrics(val_metrics, best_model_metrics, epoch)
        if found_better:
            self.log_metrics('best', epoch, best_metrics)

        return best_metrics, found_better

    def fit(self, loaders: dict):
        """
        Fit and make the model.
        Args:
            loaders: a dictionary of data loaders keyed by RunMode.
        Returns:
            Metric
        """
        logger.info(
            f"Size of datasets {dict((mode, len(loader.dataset)) for mode, loader in loaders.items())}")
        if self.resume_from_checkpoint:
            # Not sure if we need the below code
            # model_dir = self.resume_from_checkpoint
            # model_path = sorted(pathlib.Path(model_dir).glob('*pt'), key=lambda x: float(x.name.split('_')[1]))[-1]
            # self.load_checkpoint(model_path)
            self.load_checkpoint(self.resume_from_checkpoint)

        self.logging_tool.log_param('save_dir', os.getcwd())

        # Reset the counters
        self.epoch = 0
        self.resume_epoch = 0

        # Training loop
        patience = 0
        best_model_metrics = self.get_initial_model_metric()
        for epoch in range(self.resume_epoch, self.resume_epoch + 2 if self.fast_dev_run else self.max_epoch):
            best_model_metrics, found_better = self._run_epoch(epoch,
                                                               loaders[RunMode.TRAIN], loaders[RunMode.VALIDATE],
                                                               loaders.get(RunMode.TEST, None),
                                                               best_model_metrics=best_model_metrics)
            # Early Stop if patience reaches threshold.
            patience = 0 if found_better else patience + 1
            if patience >= self.early_stop_patience:
                logger.info(f"Met the early stop condition. Stopping at epoch # {epoch}.")
                break

        return best_model_metrics
    
    def test(self, loaders):
        # Test the checkpoints
        if os.path.exists(self.path_best_model):
            self.load_checkpoint(self.path_best_model)
        else:
            logger.info(
                'The best model path has never been updated, initial model has been used for testing.')

        if RunMode.TEST in loaders:
            best_model_test_metrics = self.test_epoch(self.epoch_best_model, loaders[RunMode.TEST])
            self.log_metrics('checkpoint_test', None, best_model_test_metrics)

    def log_metrics(self, run_mode_str: str, step, metrics: object, log_prefix='', mlflow_log_prefix='', duration=None):
        """ Log the metrics to logger and to mlflow if mlflow is used. Metrics could be None if learning isn't
        performed for the epoch. """
        self.logging_tool.log_metrics(run_mode_str=run_mode_str,
                                      step=step, metrics=metrics,
                                      log_prefix=log_prefix,
                                      mlflow_log_prefix=mlflow_log_prefix,
                                      duration=None)

    def log_lr(self, lr: float, step, log_prefix='', mlflow_log_prefix='', duration=None):
        """ Log the learning rate to logger and to mlflow if mlflow is used. """
        self.logging_tool.log_lr(lr, step, log_prefix, mlflow_log_prefix, duration)

class SchedulerTool:
    """
    scheduler의 종류에 따라 호출되는 방법이 다르기 때문에, 해당 툴을 활용하여 핸들링합니다.
    """

    def __init__(self, scheduler):
        # set scheduler
        self.scheduler = scheduler

        # get scheduler name
        self.scheduler_name = scheduler.__class__.__name__

        # This tool has been implemented considering the following scheduler instances.
        assert self.scheduler_name in [
            "NoneType",  # no scheduler
            "OneCycleLR",
            "CosineAnnealingWarmUpRestarts",
            "ExponentialLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau"
        ]

        # set mode between "epoch", "step", and "none".
        if self.scheduler_name == "NoneType":
            self.scheduler_mode = "none"
        elif self.scheduler_name == "OneCycleLR":
            self.scheduler_mode = "step"
        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler_mode = "epoch_val"
        else:
            self.scheduler_mode = "epoch"

    def step(self, mode="epoch", *args):
        if self.scheduler_mode == mode:
            self.scheduler.step(args[0]) if args else self.scheduler.step()
