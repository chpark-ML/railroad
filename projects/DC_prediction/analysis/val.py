import logging
import os

import hydra
import omegaconf
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import tqdm

import projects.DC_prediction.utils.constants as C
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
        logits = model(x, yaw)

        list_logits.append(logits)
        list_annots.append(y)

    preds = torch.vstack(list_logits).squeeze()
    annots = torch.vstack(list_annots).squeeze()

    return preds, annots


@hydra.main(version_base='1.2', config_path='../configs', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
    config: omegaconf.DictConfig = set_config(config)

    # load answer sheet
    df_ans = pd.read_csv(C.ANSWER_SAMPLE)

    # curved 
    ckpts = {
        'curved': C.CKPT_HOME / 'baseline/baseline-post-interval50-curved-window-4000-2000-inplane16-fmaps32/2023-08-22_23-25-54/model.pth',
        'straight': C.CKPT_HOME / 'baseline/baseline-post-interval50-straight-window-4000-2000-inplane16-fmaps32/2023-08-22_23-25-55/model.pth'
    }
    for rail in C.RAIL_TYPES:
        config.loader.dataset.rail_type=rail
        model = hydra.utils.instantiate(config.model)
        ckpt_path = ckpts[rail]

        # gpus = 0
        # torch_device = f'cuda:{gpus}'
        torch_device = f'cpu'

        if torch_device.startswith('cuda'):
            cudnn.benchmark = False
            cudnn.deterministic = True
        
        device = torch.device(torch_device)

        model = load_checkpoint(model, ckpt_path, device)

        run_modes = [RunMode('val')]
        loaders = {mode: hydra.utils.instantiate(config.loader,
                                                    dataset={'mode': mode}, shuffle=(mode == RunMode.TRAIN),
                                                    drop_last=(mode == RunMode.TRAIN)) for mode in run_modes}
        preds, _ = _inference(model, loaders[RunMode.VALIDATE], device)  # (B, 4, 2500)
        
        # TODO... check if index is correct...
        for col in C.PREDICT_COLS:
            for yaw in C.YAW_TYPES:
                target_col = f'{col}_{rail[0]}{yaw}'
                df_ans.loc[:, target_col] = preds[C.YAW_MAPPER[yaw], C.PREDICT_COL_MAPPER[col], -len(df_ans):].detach().cpu().numpy()
    
    df_ans.to_csv('/opt/railroad/projects/DC_prediction/analysis/result_val.csv', index=False)

if __name__ == '__main__':
    main()