from typing import List, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from projects.DC_prediction.utils.enums import RunMode


class RailroadDataset(Dataset):
    def __init__(self, mode: Union[str, RunMode], patch_size, dicom_window, buffer,
                 val_shift_scale, augmentation, data_path_base, mask_threshold=0.5, smoothing=0.,
                 dataset_size_scale_factor=None, target_dataset=None, dataset_info=None):
        super().__init__(mode, patch_size, dicom_window, buffer,
                         val_shift_scale, augmentation, data_path_base, mask_threshold,
                         smoothing, dataset_size_scale_factor, target_dataset, dataset_info)

    def __getitem__(self, index):
        """
        (Resize) -> Augmentation -> Windowing
        """
        elem = self.meta_df.iloc[index]
        data_source = elem['data_source']


        # Data augmentation
        if self.mode == RunMode.TRAIN:
            img = self.transform(img)

        # Data preprocessing
        img = [fn(img)[np.newaxis, ...] for fn in self.dicom_windowing]  # [(1, 48, 72, 72), ...]
        img = np.concatenate(img, axis=0)  # (n, 48, 72, 72)

        return {'x': img, 'y': None}