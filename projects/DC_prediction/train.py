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
import torch.nn.parallel as tnp
import tqdm
from sklearn import metrics
from torch.utils.data.distributed import DistributedSampler

from projects.DC_prediction.utils import experiment_tool as et
from projects.DC_prediction.utils.enums import RunMode
from projects.DC_prediction.utils.utils import (
    get_binary_classification_metrics, _seed_everything, print_config, set_config, get_torch_device_string)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Trainer')


@dataclass
class Metrics():
    loss: float = np.inf
    multi_label_losses: dict = None
    multi_label_metrics: dict = None
    multi_label_threshold: dict = None

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
                                                dataset={'mode': mode},
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
                 deterministic=True, fine_tune_info=None, is_distributed=False, is_head_rank=False,
                 early_stop_patience: int = None, use_amp: bool = True, optuna_trial: optuna.Trial = None, **kwargs) -> None:
        self.model = model
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
        self.is_distributed = is_distributed
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

        # Load pretrained encoder
        if fine_tune_info.pretrained_encoder:
            self._load_pretrained_weight(fine_tune_info.pretrained_encoder)

        self.is_head_rank = is_head_rank

        self.dict_threshold = None

    @classmethod
    def hydrate_trainer(cls, config: omegaconf.DictConfig, loaders, model, logging_tool, optuna_trial=None) -> T:
        # Init model
        logger.info(f'Instantiating model <{config.model._target_}>')

        # Init optimizer
        optimizer = hydra.utils.instantiate(config.optim, model.parameters())
        if "steps_per_epoch" in config.scheduler:
            config.scheduler["steps_per_epoch"] = len(loaders[RunMode.TRAIN])
        scheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
        criterion = hydra.utils.instantiate(config.criterion, train_df=loaders[RunMode.TRAIN].dataset.meta_df)

        # Init trainer
        logger.info(f'Instantiating trainer <{config.trainer._target_}>')
        return hydra.utils.instantiate(config.trainer,
                                       model=model,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       criterion=criterion,
                                       logging_tool=logging_tool,
                                       is_distributed=config.ddp.enable,
                                       use_amp=config.trainer.use_amp,
                                       optuna_trial=optuna_trial)

    def train_epoch(self, epoch, train_loader) -> Metrics:
        super().train_epoch(epoch, train_loader)
        train_losses = []
        start = time.time()

        train_loader.dataset.shuffle()
        for i, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            dicom = data['dicom'].to(self.device)
            _check_any_nan(dicom)
            annots = dict()
            for key, value in data['ann'].items():
                _annot = value.to(self.device).float()
                _check_any_nan(_annot)
                annots[key] = torch.unsqueeze(_annot, dim=1)

            # forward propagation
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(dicom)
                dict_loss, loss = self.criterion(logits, annots, is_logit=True, is_logistic=True)
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
                self.log_metrics(RunMode.TRAIN.value, global_step, Metrics(loss.detach(), dict_loss, None),
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

    # def get_samples_to_validate(self, dict_probs, dict_annots):
    #     for i_attr in self.target_attr_to_train:
    #         # samples where annotation label is not ambiguous, 0.5.
    #         dict_probs[i_attr] = dict_probs[i_attr][dict_annots[i_attr] != 0.5]
    #         dict_annots[i_attr] = (dict_annots[i_attr][dict_annots[i_attr] != 0.5] > 0.5) * 1.0

    #     return dict_probs, dict_annots

    def get_metrics(self, dict_probs, dict_annots):
        dict_losses, losses = self.criterion(dict_probs, dict_annots, is_logit=False, is_logistic=True)
        result_dict = get_binary_classification_metrics(dict_probs, dict_annots, self.dict_threshold)

        return losses.detach(), dict_losses, result_dict

    def _inference(self, loader):
        list_logits = []
        list_annots = []

        for data in tqdm.tqdm(loader):
            dicom = data['dicom'].to(self.device)
            annots = dict()
            for key in self.target_attr_to_train:
                value = data['ann'][key]
                annots[key] = torch.unsqueeze(value.to(self.device).float(), dim=1)
            _check_any_nan(dicom)
            logits = self.model(dicom)

            list_logits.append(logits)
            list_annots.append(annots)

            if self.fast_dev_run:
                # Runs 1 train batch and program ends if 'fast_dev_run' set to 'True'
                # TODO: runs 'n(int)' train batches otherwise.
                # FIXME: progress bar does not update when 'fast_dev_run==True'
                break

        dict_probs = {key: torch.vstack([torch.sigmoid(i_logits[key]) for i_logits in list_logits]).squeeze() for key in
                      self.target_attr_to_train}
        dict_annots = {key: torch.vstack([i_annots[key] for i_annots in list_annots]).squeeze() for key in
                       self.target_attr_to_train}

        dict_probs, dict_annots = self.get_samples_to_validate(dict_probs, dict_annots)

        return dict_probs, dict_annots

    @torch.no_grad()
    def validate_epoch(self, epoch, val_loader) -> Metrics:
        super().validate_epoch(epoch, val_loader)
        dict_probs, dict_annots = self._inference(val_loader)
        loss, dict_loss, dict_metrics = self.get_metrics(dict_probs, dict_annots)
        self.scheduler.step("epoch_val", loss)

        return Metrics(loss, dict_loss, dict_metrics, self.dict_threshold)

    @torch.no_grad()
    def test_epoch(self, epoch, test_loader) -> Metrics:
        super().test_epoch(epoch, test_loader)
        dict_probs, dict_annots = self._inference(test_loader)
        loss, dict_loss, dict_metrics = self.get_metrics(dict_probs, dict_annots)

        return Metrics(loss, dict_loss, dict_metrics, self.dict_threshold)

    def save_best_metrics(self, val_metrics: Metrics, best_metrics: Metrics, epoch) -> (
            object, bool):
        found_better = False
        if val_metrics.loss < best_metrics.loss:
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

        for key, value in self.dict_threshold.items():
            checkpoint[key] = value

        if not self.is_distributed or (self.is_distributed and self.is_head_rank):
            # Save when
            # 1. Experiments without DDP save the checkpoint as usual.
            # 2. Use DDP and should be head rank
            torch.save(checkpoint, model_path)


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
