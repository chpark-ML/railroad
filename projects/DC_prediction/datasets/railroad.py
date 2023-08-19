from typing import Union, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from projects.DC_prediction.utils.enums import RunMode
import projects.DC_prediction.utils.constants as C


def _get_start_distance(mode: RunMode):
    pass


def _get_df() -> Dict[str, Dict[str, pd.DataFrame]]:
    dict_df_rail = dict()  # {'curved': {'30': df, '40': df, ...},
                           #  'straight': {'30': df, '40': df, ...} }
    for rail in C.RAIL_TYPES:  
        drive_files = C.VIBRATION_DATA_CURVED if rail == "curved" else C.VIBRATION_DATA_STRAIGHT
        df_lane = pd.read_csv(C.LANE_DATA_CURVED if rail == "curved" else C.LANE_DATA_STRAIGHT)
        df_lane = df_lane.set_index(keys=['Distance'], inplace=False)
        
        dict_df_yaw = dict()
        for _dir in drive_files:
            df_vib = pd.read_csv(_dir).set_index(keys=['Distance'], inplace=False)
            df_concat = pd.concat([df_vib, df_lane], axis=1).sort_index(ascending=True)
            yaw = _dir.stem.split("_")[-1][1:]
            dict_df_yaw[yaw] = df_concat
        dict_df_rail[rail] = dict_df_yaw
    return dict_df_rail


class RailroadDataset(Dataset):
    def __init__(self, mode: Union[str, RunMode], window_length, history_length):
        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        self.window_length = window_length
        self.history_length = history_length
        self.df = _get_start_distance(self.mode) # TODO: # row should be corresponding to a starting point of a sample window (start distance info, rail type, yaw type)
        self.dfs = _get_df()  # dictionary of dictionary for (rail type, yaw type)

    def __getitem__(self, index: int):
        """
        Resize -> Windowing -> Additoinal data augmentation
        """
        elem = self.meta_df.iloc[index]
        data_source = elem['data_source']

        # TODO: resize/interpolation of input signal
        
        # TODO: windowing

        # Data augmentation
        if self.mode == RunMode.TRAIN:
            img = self.transform(img)

        # Data preprocessing
        img = [fn(img)[np.newaxis, ...] for fn in self.dicom_windowing]  # [(1, 48, 72, 72), ...]
        img = np.concatenate(img, axis=0)  # (n, 48, 72, 72)

        return {'x': img, 'y': None}
