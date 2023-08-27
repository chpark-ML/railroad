import random
from typing import Union, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from projects.DC_prediction.utils.enums import RunMode
import projects.DC_prediction.utils.constants as C
import projects.DC_prediction.utils.augmentation as aug


def _get_start_distance(mode: RunMode,
                        val_type: str,
                        window_length: int,
                        history_length: int,
                        test_input_length: int,
                        rail_type: str = "curved",
                        interval: int = 50) -> pd.DataFrame:
    assert rail_type in C.RAIL_TYPES_TO_TRAIN
    data = {'start_distance': [], 'rail_type': [], 'yaw_type': []}
    if interval is None:
        interval = 1

    if rail_type == 'both':
        rail_type_list = C.RAIL_TYPES
    else:
        rail_type_list = [rail_type]

    for rail in rail_type_list:
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
                elif val_type == 'wo':
                    # e.g., [0,...,7500]
                    # window (0~2499), ... ,(7500~9999)
                    list_start_distance = list(range(0,
                                                    C.PREDICT_START_INDEX - window_length + 1,
                                                    interval))  

            elif mode == RunMode.VALIDATE:
                if val_type == 'pre':
                    # e.g, [0], window (0~2499)
                    list_start_distance = [0]
                elif val_type == 'post':
                    # e.g, [7500], window (7500 ~ 9999)
                    list_start_distance = [C.PREDICT_START_INDEX - window_length]
                elif val_type == 'wo':
                    # e.g, []
                    list_start_distance = []

            elif mode == RunMode.TEST:
                # e.g., [9500] / window (9500 ~ 11999)
                list_start_distance = [C.PREDICT_START_INDEX - history_length] 

            data['start_distance'].extend(list_start_distance)
            data['rail_type'].extend([rail] * len(list_start_distance))
            data['yaw_type'].extend([yaw] * len(list_start_distance))

    df = pd.DataFrame(data)
    return df


def _get_df(use_df_lane=True) -> Dict[str, Dict[str, pd.DataFrame]]:
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

            if use_df_lane:
                df_concat = pd.concat([df_vib, df_lane], axis=1).sort_index(ascending=True)
            else:
                df_concat = df_vib.sort_index(ascending=True)
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

    def __call__(self, x, y):
        if self.mode == 'train':
            # Apply transform
            for f in self.transform:
                x, y = f(x, y)

        return x, y
        

class RailroadDataset(Dataset):
    def __init__(self,
                 mode: Union[str, RunMode],
                 val_type: str,
                 window_length: int,
                 history_length: int,
                 rail_type: str = "curved",
                 interval: int = None, 
                 augmentation: dict = None):
        assert val_type in ["pre", "post", "wo"]
        assert history_length < window_length
        assert rail_type in C.RAIL_TYPES_TO_TRAIN

        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        self.val_type = val_type
        self.window_length = window_length 
        self.history_length = history_length
        self.test_input_length = window_length
        if self.mode == RunMode.TEST and self.test_input_length < C.PREDICT_LENGHT:
            self.test_input_length = C.PREDICT_LENGHT + self.history_length
            assert self.test_input_length % 2 == 0
            
        self.interval = interval
        self.num_channels = C.NUM_CHANNEL_MAPPER[rail_type]

        self.df_data = _get_df(use_df_lane=(rail_type!='both'))  # dictionary of dictionary for (rail type, yaw type)
        self.df_index = _get_start_distance(mode=self.mode, val_type=self.val_type,
                                            window_length=self.window_length,
                                            history_length=self.history_length,
                                            test_input_length=self.test_input_length,
                                            rail_type=rail_type, interval=interval)

        if self.mode == RunMode.TRAIN:
            self.transform = Compose(transform=[
                aug.GaussianSmoothing(p=augmentation['gaussian_smoothing']['p'],
                                      num_channels=self.num_channels,
                                      mode=augmentation['gaussian_smoothing']['mode'],
                                      min_sigma=augmentation['gaussian_smoothing']['min_sigma'],
                                      max_sigma=augmentation['gaussian_smoothing']['max_sigma'],
                                      sigma_normal_scale=augmentation['gaussian_smoothing']['sigma_normal_scale']),
                aug.RescaleTime(p=augmentation['rescale_time']['p'], 
                                min_scale_factor=augmentation['rescale_time']['min_scale_factor'], 
                                max_scale_factor=augmentation['rescale_time']['max_scale_factor']),
            ])
    
    def __len__(self):
        return len(self.df_index)
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, int]]:
        """
        Resize -> Windowing -> Additional data augmentation
        """
        elem = self.df_index.iloc[index]
    
        sd = elem['start_distance']
        if self.interval is not None and self.mode == RunMode.TRAIN:
            # Add stochasticity to starting distance.
            # This should be controlled by another parameter, instead of directly using `interval`
            # TODO: results in different tensor size across the batch sample.
            max_shift_size = int(self.interval * 1.0)
            sd += np.random.randint(low=0, high=max_shift_size)
            sd = max(sd, self.df_index['start_distance'].max())
        rail = elem['rail_type']
        yaw = elem['yaw_type']
        data = self.df_data[rail][yaw]

        if self.mode == RunMode.TRAIN or self.mode == RunMode.VALIDATE:
            x = data.loc[sd : sd+self.window_length-1].copy()
            y = data.loc[sd : sd+self.window_length-1].loc[:, C.PREDICT_COLS].copy()
            for col in C.PREDICT_COLS:
                x.loc[self.history_length if self.mode == RunMode.VALIDATE else np.random.uniform(0, self.history_length):, col] = 0
        elif self.mode == RunMode.TEST:
            x = data.loc[sd : ].copy()
            y = data.loc[sd : ].loc[:, C.PREDICT_COLS].copy()

        x = np.transpose(x.values, axes=(1, 0)) # (channel, time)
        y = np.transpose(y.values, axes=(1, 0)) # (4, time)

        # Data augmentation
        if self.mode == RunMode.TRAIN:
            x, y = self.transform(x, y)

        x = np.expand_dims(x, axis=(0)) # (1, channel, time)
        y = np.expand_dims(y, axis=(0)) # (1, 4, time)

        return {'x': x.astype('float32'),
                'y': y.astype('float32'),
                "yaw": C.YAW_MAPPER[yaw]}
