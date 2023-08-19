import os
from pathlib import Path

YAW_TYPES = ['30', '40', '50', '70', '100']
RAIL_TYPES = ['curved', 'straight']
PREDICT_COLS = ["YL_M1_B1_W1", "YR_M1_B1_W1", "YL_M1_B1_W2", "YR_M1_B1_W2"]

DATA_ROOT_PATH = Path('./data')

VIBRATION_DATA_STRAIGHT = sorted(DATA_ROOT_PATH.rglob("*data_s*0.csv"))
LANE_DATA_STRAIGHT = DATA_ROOT_PATH / "lane_data_s.csv"

VIBRATION_DATA_CURVED = sorted(DATA_ROOT_PATH.rglob("*data_c*0.csv"))
LANE_DATA_CURVED = DATA_ROOT_PATH / "lane_data_c.csv"