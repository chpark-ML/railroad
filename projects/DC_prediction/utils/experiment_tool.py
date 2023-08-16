""" BaseExperimentToolLogger for various experiment tools.
Currently, supports Mlflow and AimStack
"""
import logging
import os
from abc import abstractmethod
from typing import Optional, Sequence, Union

import hydra
import mlflow
import omegaconf
import pandas as pd
from pathlib import Path

from projects.DC_prediction.utils import utils

logger = logging.getLogger(__name__)


def _log_params_from_omegaconf_dict(params: omegaconf.dictconfig.DictConfig, logging_tool):
    def _explore_recursive(parent_name, element):
        if isinstance(element, omegaconf.DictConfig):
            for k, v in element.items():
                if isinstance(v, omegaconf.DictConfig) or isinstance(v, omegaconf.ListConfig):
                    _explore_recursive(f'{parent_name}.{k}', v)
                else:
                    logging_tool.log_param(f'{parent_name}.{k}', v)
        elif isinstance(element, omegaconf.ListConfig):
            for i, v in enumerate(element):
                logging_tool.log_param(f'{parent_name}.{i}', v)
        else:
            pass  # recursive terminal

    for param_name, param_value in params.items():
        _explore_recursive(param_name, param_value)


class BaseExperimentToolLogger:
    """ Base Experiment tool logger that contains abstract methods.
    This will also work as empty logger that does not do any actions when no logger is successfully initiated.
    Takes following style of yaml
    ```yaml
    experiment_tool:
        client: # This is optional. mlflow/wandb does not need a client, while aimstack uses `Run` object to track metrics
            _target_: ...
        name: mlflow # Feed logger name (mlflow, aim)
        enable: True # This is required for DDP environs.
        experiment_name: lct-cls # Large pool that contains all runs
        run_name: AdamW lr=1e-4  # Single run name. Collection of runs is experiment.
    ```
    """

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        self._initiate(config)

    @abstractmethod
    def _initiate(self, config: omegaconf.dictconfig.DictConfig):
        pass

    @abstractmethod
    def log_param(self, key, value):
        """ Parameters for the experiment (e.g. hyperparameters, gpu_num, initialized learning rate)"""

    @abstractmethod
    def log_metrics(self, run_mode_str: str, step, metrics: object, log_prefix="", mlflow_log_prefix="", duration=None):
        """ Metrics being tracked along training (e.g. loss, accuracy) """
        log_str = f'{log_prefix} {run_mode_str} {metrics}'
        if duration:
            log_str += f' time:{duration:.4f}'
        logger.info(f"{log_str} step:{step}")

    @abstractmethod
    def log_lr(self, lr, step, log_prefix='', mlflow_log_prefix="", duration=None):
        """ Logging learning rate """
        log_str = f'{log_prefix} {lr}'
        if duration:
            log_str += f' time:{duration:.4f}'
        logger.info(f"{log_str} step:{step}")

    @abstractmethod
    def raise_error(self, erorr: Exception):
        """ Actions when error is raised (e.g. leave tags for failure cases) """

    @abstractmethod
    def raise_keyboardinterrupt(self, error: Exception = KeyboardInterrupt):
        """ Same purpose with `raise_error` method but for 'intentionally' killed experiments. """

    @abstractmethod
    def log_model(self, pytorch_model, artifact_path: str):
        """ Saves model checkpoint artifacts"""

    def end_run(self, status="FINISHED", best_model_metrics=None):
        """ Define a behavior when experiment is done. May not be necessary for some tools """

    @abstractmethod
    def log_image(self, image=None, context=None, step: int = 0, title: str = ""):
        """ Log image """


class MlflowLogger(BaseExperimentToolLogger):
    name = "mlflow"

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__(config)
        _log_params_from_omegaconf_dict(config, self)
        self.run_status = None

    def _initiate(self, config: omegaconf.dictconfig.DictConfig):
        # Wait for 30 minutes in case there is an outage
        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = str(60 * 30)  # 30 minutes. Default is 120.
        exp_config = config.experiment_tool
        mlflow.set_tracking_uri(exp_config.server_uri)
        mlflow.set_experiment(exp_config.experiment_name)
        mlflow.start_run(run_name=exp_config.get("run_name", None))
        mlflow.set_tag("host_ip", utils.get_host_ip_address())

    def log_model(self, pytorch_model, artifact_path: str):
        try:
            model_info = mlflow.pytorch.log_model(pytorch_model=pytorch_model, artifact_path=artifact_path)
            logger.info(f"mlflow logged model info is: {model_info}")
        except PermissionError as e:
            logger.warning(f"Was not able to save the model due to permission error.")
            logger.exception(e)

    def log_metrics(self, run_mode_str: str, step, metrics: Union[object, dict], log_prefix="", mlflow_log_prefix="",
                    duration=None):
        super().log_metrics(run_mode_str, step, metrics, log_prefix, mlflow_log_prefix, duration)
        if metrics:
            # transform object to dict type
            if isinstance(metrics, object) and not isinstance(metrics, dict):
                metrics = vars(metrics)

            dict_metrics = dict()
            for key, value in metrics.items():
                if not isinstance(value, dict):
                    dict_metrics[key] = value
                else:
                    for k, v in value.items():
                        dict_metrics[k] = v
            metrics = dict_metrics

            # if checkpoint validation, save into dataframe type for further analysis
            if 'checkpoint_' in run_mode_str:
                df = pd.DataFrame.from_dict(data=[metrics])
                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                save_dir = Path(hydra_cfg['runtime']['output_dir'])
                df.to_csv(save_dir / (run_mode_str + '.csv'), index=False)

            # mlflow logging
            if mlflow_log_prefix:
                metrics_to_log = {f'{mlflow_log_prefix}-{run_mode_str}-{k}': v for k, v in metrics.items() if
                                  v is not None}
            else:
                metrics_to_log = {f'{run_mode_str}-{k}': v for k, v in metrics.items() if v is not None}
            mlflow.log_metrics(metrics_to_log, step)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_lr(self, lr: float, step, log_prefix='', mlflow_log_prefix='', duration=None):
        super().log_lr(lr, step, log_prefix, mlflow_log_prefix)
        metrics_key = f'{mlflow_log_prefix}-lr' if mlflow_log_prefix else 'lr'
        mlflow.log_metrics({metrics_key: lr}, step)

    def set_tag(self, key, value):
        mlflow.set_tag(key, value)

    def raise_error(self, error: Exception):
        self.set_tag(key="FAIL_REASON", value=error)
        self.run_status = "FAILED"

    def raise_keyboardinterrupt(self, error: Exception = KeyboardInterrupt):
        self.raise_error(error)
        self.run_status = "KILLED"

    def end_run(self, status="FINISHED"):
        # Former has higher priority in string compare or condition.
        # e.g. self.run_status="KILLED", status="FINISHED"
        # then status = self.run_status or status will assign "KILLED"
        status = self.run_status or status
        mlflow.end_run(status=status)

    def log_image(self, image=None, context=None, step: int = 0, title: str = ""):
        raise NotImplemented()


def load_logging_tool(config: omegaconf.dictconfig.DictConfig):
    if config.get("experiment_tool") and config.get("experiment_tool").get("enable"):
        name = config.experiment_tool.name
        try:
            logging_tool = {"mlflow": MlflowLogger}[name](config=config)
            logger.info("%s is used.", logging_tool.name)
        except Exception as e:
            # Expected to return empty BaseExperimentLogger
            logger.warning("No available logging tool for %s. Default logger will be used.", name)
            logger.exception(e)

    else:
        logging_tool = BaseExperimentToolLogger(config=config)

    return logging_tool