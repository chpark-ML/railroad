import logging
import os
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf

import projects.DC_prediction.utils.constants as C
import projects.DC_prediction.utils.model_path as MP
from projects.DC_prediction.train import _get_loaders_and_trainer
from projects.DC_prediction.utils.enums import RunMode
from projects.DC_prediction.utils.utils import set_config
from projects.DC_prediction.utils.utils import (
    get_binary_classification_metrics, _seed_everything, print_config, set_config, get_torch_device_string)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(model, ckpt_path, device):
    """Loads checkpoint from directory"""
    assert os.path.exists(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    logger.info(f'Model loaded from {ckpt_path}')

    return model


def _inference(model, loader, device):
    list_logits = []
    list_annots = []
    model.eval()

    for data in tqdm.tqdm(loader):
        x = data['x'].to(device)
        y = data['y'].to(device)
        yaw = data['yaw'].to(device)
        
        if loader.dataset.test_input_length == loader.dataset.window_length:
            logits = model(x, yaw)
        else:
            _interval = loader.dataset.window_length // 2
            _num_windowing = 1 + int(np.ceil((x.size(3) - loader.dataset.window_length) / _interval))
            _expected_size = loader.dataset.window_length + int(np.ceil((x.size(3) - loader.dataset.window_length) / _interval)) * _interval
            diff_t = _expected_size - loader.dataset.test_input_length
            x_padded = F.pad(x, [0, diff_t, 0, 0])
            logits = torch.zeros((x.size(0), 1, len(C.PREDICT_COLS), x.size(3)))
            counters = torch.zeros((x.size(0), 1, len(C.PREDICT_COLS), x.size(3)))
            for i in range(_num_windowing):
                start_index = i * _interval
                end_index = start_index + loader.dataset.window_length 
                logits_padded = model(x_padded[:, :, :, start_index: end_index], yaw)
                logits[:, :, :, start_index: end_index] = logits_padded
                counters[:, :, :, start_index: end_index] += 1
            logits = (logits / counters)[:, :, :, :x.size(3)]

        list_logits.append(logits)
        list_annots.append(y)

    preds = torch.vstack(list_logits)
    annots = torch.vstack(list_annots)

    return preds, annots


def main() -> None:
    # load answer sheet
    df_ans = pd.read_csv(C.ANSWER_SAMPLE)

    # checkpoint path
    ckpts = MP.TARGET_MODEL_PATH

    for rail in C.RAIL_TYPES:
        ckpt_path = ckpts[rail] / Path("model.pth")
        config_path = ckpts[rail] / Path(".hydra/config.yaml")
        config = OmegaConf.load(config_path)
        model = hydra.utils.instantiate(config.model)
        

        # gpus = 0
        # torch_device = f'cuda:{gpus}'
        torch_device = f'cpu'

        if torch_device.startswith('cuda'):
            cudnn.benchmark = False
            cudnn.deterministic = True
        
        device = torch.device(torch_device)

        model = load_checkpoint(model, ckpt_path, device)

        run_modes = [RunMode('test')]
        loaders = {mode: hydra.utils.instantiate(config.loader,
                                                dataset={'mode': mode}, shuffle=(mode == RunMode.TRAIN),
                                                drop_last=(mode == RunMode.TRAIN)) for mode in run_modes}
        
        preds, _ = _inference(model, loaders[RunMode.TEST], device)  # (B, 4, 2500)
        
        assert preds.size(0) == 5 or preds.size(0) == 10
        if preds.size(0) == 10:
            print('both type!')
            for rail in C.RAIL_TYPES:
                _preds = preds[C.RAIL_MAPPER[rail] * 5 : (C.RAIL_MAPPER[rail]+1) * 5]
                for col in C.PREDICT_COLS:
                    for yaw in C.YAW_TYPES:
                        target_col = f'{col}_{rail[0]}{yaw}'
                        df_ans.loc[:, target_col] = _preds[C.YAW_MAPPER[yaw], 0, C.PREDICT_COL_MAPPER[col], -len(df_ans):].detach().cpu().numpy()
            break

        else:
            for col in C.PREDICT_COLS:
                for yaw in C.YAW_TYPES:
                    target_col = f'{col}_{rail[0]}{yaw}'
                    df_ans.loc[:, target_col] = preds[C.YAW_MAPPER[yaw], 0, C.PREDICT_COL_MAPPER[col], -len(df_ans):].detach().cpu().numpy()
    
    df_ans.to_csv('/opt/railroad/projects/DC_prediction/analysis/result_test.csv', index=False)

if __name__ == '__main__':
    main()
