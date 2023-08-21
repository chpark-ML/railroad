from typing import Union, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from projects.DC_prediction.utils.enums import RunMode
import projects.DC_prediction.utils.constants as C


def _get_start_distance(mode: RunMode,
                        val_type: str,
                        window_length: int,
                        history_length: int,
                        rail_type: str = "curved",
                        interval: int = 50) -> pd.DataFrame:
    assert rail_type in C.RAIL_TYPES
    data = {'start_distance': [], 'rail_type': [], 'yaw_type': []}
    if interval is None:
        interval = 1

    # 아래 예시 주석은 다음 변수일 때 기대값입니다. (window length: 2500, histroy_length: 500)
    for yaw in C.YAW_TYPES:
        if mode == RunMode.TRAIN:
            if val_type == 'pre':
                # e.g., [2500,...,7500]
                # window (2500~4999), ... ,(7500~9999)
                list_start_distance = list(range(window_length,
                                                 C.PREDICT_START_INDEX - window_length + 1,
                                                 interval))
            elif val_type == 'post':
                # e.g., [0,...,5500]
                # window (0~2499), ... ,(5500~7999)
                list_start_distance = list(range(0,
                                                 C.PREDICT_START_INDEX - window_length + history_length - window_length + 1,
                                                 interval))  

        elif mode == RunMode.VALIDATE:
            if val_type == 'pre':
                # e.g, [0], window (0~2499)
                list_start_distance = [0]
            elif val_type == 'post':
                # e.g, [7500], window (7500 ~ 9999)
                list_start_distance = [C.PREDICT_START_INDEX - window_length]  

        elif mode == RunMode.TEST:
            # e.g., [9500] / window (9500 ~ 11999)
            list_start_distance = [C.PREDICT_START_INDEX - history_length] 

        data['start_distance'].extend(list_start_distance)
        data['rail_type'].extend([rail_type] * len(list_start_distance))
        data['yaw_type'].extend([yaw] * len(list_start_distance))

    df = pd.DataFrame(data)
    return df


def _get_df() -> Dict[str, Dict[str, pd.DataFrame]]:
    """ Loads all `.csv` from data path"""
    dict_df_rail = dict()  # {'curved': {'30': df, '40': df, ...},
                           #  'straight': {'30': df, '40': df, ...} }
    for rail in C.RAIL_TYPES:  
        drive_files = C.VIBRATION_DATA_CURVED if rail == "curved" else C.VIBRATION_DATA_STRAIGHT
        df_lane = pd.read_csv(C.LANE_DATA_CURVED if rail == "curved" else C.LANE_DATA_STRAIGHT)
        assert all(df_lane['Distance'].values * 4 == (df_lane['Distance'].values * 4).astype(int))
        df_lane['Distance'] = (df_lane['Distance'] * 4).astype(int) # [0.00 ~ 2500.00] / [2500.25 ~ 2999.75] > [0 ~ 10000] / [10001 ~ 11999]
        df_lane = df_lane.set_index(keys=['Distance'], inplace=False)

        dict_df_yaw = dict()
        for _dir in drive_files:
            df_vib = pd.read_csv(_dir)
            assert all(df_vib['Distance'].values * 4 == (df_vib['Distance'].values * 4).astype(int))
            df_vib['Distance'] = (df_vib['Distance'] * 4).astype(int)
            df_vib = df_vib.set_index(keys=['Distance'], inplace=False)

            df_concat = pd.concat([df_vib, df_lane], axis=1).sort_index(ascending=True)
            yaw = _dir.stem.split("_")[-1][1:]
            dict_df_yaw[yaw] = df_concat
        dict_df_rail[rail] = dict_df_yaw
    return dict_df_rail


class Compose:
    def __init__(self, transform=None, mode: Union[str, RunMode] = 'train'):
        assert mode in ['train', 'val', 'test']
        if mode != 'train':
            assert transform
        self.mode = mode
        self.transform = transform

    def __call__(self, tensor, mask=None):
        if self.mode == 'train':
            # Apply transform
            for f in self.transform:
                tensor = f(tensor)

        return tensor
        

class RailroadDataset(Dataset):
    def __init__(self,
                 mode: Union[str, RunMode],
                 val_type: str,
                 window_length: int,
                 history_length: int,
                 rail_type: str = "curved",
                 interval: int = None):
        assert val_type in ["pre", "post"]
        assert window_length > 500
        assert history_length < window_length
        assert rail_type in C.RAIL_TYPES

        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        self.val_type = val_type
        self.window_length = window_length
        self.history_length = history_length
        self.interval = interval

        self.df_data = _get_df()  # dictionary of dictionary for (rail type, yaw type)
        self.df_index = _get_start_distance(mode=self.mode, val_type=self.val_type,
                                            window_length=self.window_length,
                                            history_length=self.history_length,
                                            rail_type=rail_type, interval=interval)

        if self.mode == RunMode.TRAIN:
            self.transform = Compose(transform=[])
    
    def __len__(self):
        return len(self.df_index)
    
    def __getitem__(self, index: int) -> Dict[str, np.ndarray | int]:
        """
        Resize -> Windowing -> Additional data augmentation
        """
        elem = self.df_index.iloc[index]
    
        sd = elem['start_distance']
        if self.interval is not None:
            # Add stochasticity to starting distance.
            # This should be controlled by another parameter, instead of directly using `interval`
            w = max(1, int(self.interval * 0.05))
            sd += np.random.randint(low=-w, high=w)
        rail = elem['rail_type']
        yaw = elem['yaw_type']
        data = self.df_data[rail][yaw]

        x = data.loc[sd : sd+self.window_length-1].to_numpy(copy=True)
        y = data.loc[sd : sd+self.window_length-1].loc[:, C.PREDICT_COLS].to_numpy(copy=True)
        for col in C.PREDICT_COLS:
            x.loc[self.history_length:, col] = 0

        x = np.transpose(np.expand_dims(x, axis=(0)), axes=(0, 2, 1)) # (1, channel, time)
        y = np.transpose(np.expand_dims(y, axis=(0)), axes=(0, 2, 1)) # (1, 4, time)
        # Data augmentation
        if self.mode == RunMode.TRAIN:
            x = self.transform(x)

        return {'x': x.astype('float32'),
                'y': y.astype('float32'),
                "yaw": C.YAW_MAPPER[yaw]}
