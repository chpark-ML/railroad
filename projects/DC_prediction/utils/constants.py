import os

from glob import glob
from enum import Enum

YAW_TYPES = ['30', '40', '50', '70', '100']
RAIL_TYPES = ['curved', 'straight']

DATA_ROOT_PATH = '/data/railroad/data'

VIBRATION_DATA_STRAIGHT = glob(os.path.join(DATA_ROOT_PATH, '/data_s**0.csv'))  # [data_s30.csv, ...]
LANE_DATA_STRAIGHT = os.path.join(DATA_ROOT_PATH, '/lane_data_s.csv')

VIBRATION_DATA_CURVED = glob(os.path.join(DATA_ROOT_PATH, '/data_c**0.csv'))  # [data_c30.csv, ...]
LANE_DATA_CURVED = os.path.join(DATA_ROOT_PATH, '/lane_data_c.csv')
