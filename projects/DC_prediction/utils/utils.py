import importlib
import logging

import GPUtil
import hydra
import os
import random
import re
import socket
import torch
import torch.nn.parallel as tnp
import rich.syntax
import rich.tree
import warnings
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.style import Style
from sklearn import metrics
from torch.utils.data.distributed import DistributedSampler

import projects.DC_prediction.utils.experiment_tool as et
from projects.DC_prediction.utils.enums import RunMode


_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_CONFIG_FILE = os.path.join(_THIS_DIR, '..', 'configs', 'config.yaml')
logger = logging.getLogger(__name__)


def get_host_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_config(config: OmegaConf, default_config_path: str = _DEFAULT_CONFIG_FILE) -> OmegaConf:
    """
    Applies optional utilities, controlled by main config file:
    - disabling warnings
    - set debug-friendly mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        default_config_path (str): path of the default config to base on.
    """
    if default_config_path:
        config = OmegaConf.merge(OmegaConf.load(default_config_path), config)

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Disable python warnings if <config.ignore_warnings=True>
    if config.get('ignore_warnings'):
        logger.info("Disabling python warnings <config.ignore_warnings=True>")
        warnings.filterwarnings('ignore')

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get('debug'):
        logger.info("Running in debug mode <config.debug=True>")
        config.trainer.fast_dev_run = True
        config.experiment_tool.enable = False

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get('fast_dev_run'):
        if config.trainer.get('gpus'):
            # If env var is set, use the one that was chosen, and set the gpu number to 0
            device_id = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            config.trainer.gpus = 0 if device_id else _get_available_gpu()

    # if auto is set for trainer's gpu, assign it here
    if config.trainer.get('gpus') == 'auto':
        config.trainer.gpus = _get_available_gpu()

    return config


def print_config(config: DictConfig,
                 resolve: bool = True) -> None:
    """
    Prints the content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    style = Style(color="white", bgcolor="black")
    tree = rich.tree.Tree(':gear: CONFIG', style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)


def _get_available_gpu():
    device_ids = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)
    return device_ids[0] if device_ids else 'cpu'


def get_binary_classification_metrics(prob: dict, annot: dict, threshold: dict):
    assert type(prob) == type(annot)
    result_dict = dict()
    target_attr = list(prob.keys())
    for i_attr in target_attr:
        result_dict[f'acc_{i_attr}'] = metrics.accuracy_score(annot[i_attr].squeeze().cpu().numpy(),
                                                              prob[i_attr].squeeze().cpu().numpy() > threshold[
                                                                  f'youden_{i_attr}'])
        try:
            result_dict[f'auroc_{i_attr}'] = metrics.roc_auc_score(annot[i_attr].squeeze().cpu().numpy(),
                                                                   prob[i_attr].squeeze().cpu().numpy())
        except ValueError:  # in the case when only one class exists, AUROC can not be calculated. (in fast_dev_run)
            pass
        result_dict[f'f1_{i_attr}'] = metrics.f1_score(annot[i_attr].squeeze().cpu().numpy(),
                                                       prob[i_attr].squeeze().cpu().numpy() > threshold[
                                                           f'youden_{i_attr}'])

    return result_dict


def get_torch_device_string(gpus):
    """
    Get torch device string based on GPU config.
    Args:
        gpus: one of ['auto', 'None', ','.join(\\d+)] otherwise an error will be raised.
            'auto': Select the first available GPU. If there is no available GPU, CPU setting will return.
            'None': 'cpu' is returned.
            0[,1,2,3]: a list of 'cuda:\\d' is returned.
    Returns:
        device string value that is to call torch.device(str) or a list of device strings in case multiple values are
         set.
    """
    if gpus == 'auto':
        # GPU 사양에 따라 maxLoad, maxMemory의 적정값이 바뀔 수 있음.
        # GPUtil계열 package는 신형 GPU가 나올때마다 호환성 이슈가 빈번히 발생: 3090, 3080, RTX TITAN 세 군데에서 작동함,
        # 향후 새로운 GPU가 추가되면 (40xx?) 에러가 날 수 있음
        device_ids = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)
        return f'cuda:{device_ids[0]}' if device_ids else 'cpu'
    elif isinstance(gpus, int):
        return f'cuda:{gpus}'
    elif gpus is None or gpus in ('None', 'cpu'):
        return 'cpu'
    elif re.match('^\\d(,\\d)*$', gpus):
        return [f'cuda:{num}' for num in gpus.split(',')]
    else:
        raise ValueError(f"Valid values for gpus is 'auto', 'None', or comma separated digits. invalid: {gpus}")
