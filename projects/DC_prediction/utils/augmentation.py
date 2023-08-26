import random 

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

import projects.DC_prediction.utils.constants as C


class GaussianSmoothing:
    def __init__(self, p: float = 0.5, num_channels: int = 32, sigma_normal_scale: float = 0.5):
        assert 0. <= p <= 1.
        self.p = p
        self.num_channels = num_channels
        self.sigma_normal_scale = sigma_normal_scale

    def __call__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 2

        if random.random() <= self.p:
            smoothed_data = np.zeros_like(x)
            smoothed_y = np.zeros_like(y)
            _sigma=np.abs(np.random.normal(0, self.sigma_normal_scale))
            for channel in range(self.num_channels):
                smoothed_data[channel] = gaussian_filter1d(x[channel], sigma=_sigma)
            for channel in range(len(C.PREDICT_COLS)):
                smoothed_y[channel] = gaussian_filter1d(y[channel], sigma=_sigma)
        else:
            smoothed_data = x
            smoothed_y = y

        return smoothed_data, smoothed_y
    

class RescaleTime:
    def __init__(self, p: float = 0.5, min_scale_factor: float = 0.9, max_scale_factor: float = 1.1):
        assert 0. <= p <= 1.
        self.p = p
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

    def __call__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 2

        if random.random() <= self.p:
            # random rescale
            random_scale_factor = tuple([1, float(np.random.uniform(low=self.min_scale_factor, high=self.max_scale_factor, size=1))])
            _x = F.interpolate(torch.tensor(x).unsqueeze(0).unsqueeze(0), scale_factor=random_scale_factor, 
                               mode="bilinear", align_corners=True)
            _y = F.interpolate(torch.tensor(y).unsqueeze(0).unsqueeze(0), scale_factor=random_scale_factor,
                               mode="bilinear", align_corners=True)
            # get padding size
            input_shape = x.shape
            rescaled_input_shape = np.array(_x.shape[-2:])
            diff = input_shape - rescaled_input_shape
            
            # resize 이전 크기와 비교했을 떄의 차이는 과거 데이터를 지우는 것으로 한다.
            pad_left_only = tuple(np.array([[0, i] for i in diff]).ravel()[::-1])

            # zero padding
            _x = F.pad(_x, pad_left_only, "constant", 0).squeeze().numpy()
            _y = F.pad(_y, pad_left_only, "constant", 0).squeeze().numpy()
        else:
            _x = x
            _y = y

        return _x, _y
