import os
from pathlib import Path

YAW_TYPES = ['30', '40', '50', '70', '100']
YAW_MAPPER = {yaw: idx for idx, yaw in enumerate(YAW_TYPES)}
RAIL_TYPES = ['curved', 'straight']
RAIL_MAPPER = {rail: idx for idx, rail in enumerate(RAIL_TYPES)}
RAIL_TYPES_TO_TRAIN = ['curved', 'straight', 'both']
NUM_CHANNEL_MAPPER = {'curved': 37, 'straight': 39, 'both': 34}

PREDICT_COLS = ["YL_M1_B1_W1", "YR_M1_B1_W1", "YL_M1_B1_W2", "YR_M1_B1_W2"]
PREDICT_COL_MAPPER = {col: idx for idx, col in enumerate(PREDICT_COLS)}
PREDICT_START_INDEX = 10000     # 제공된 데이터는 10001부터 11999까지 가려져있지만, 2000(10000~11999) 타임포인트 예측하는 모델로 정의)
PREDICT_START_DISTANCE = 2500   
PREDICT_LENGHT = 2000

DATA_ROOT_PATH = Path('/opt/railroad/data')

VIBRATION_DATA_STRAIGHT = sorted(DATA_ROOT_PATH.rglob("*data_s*0.csv"))
LANE_DATA_STRAIGHT = DATA_ROOT_PATH / "lane_data_s.csv"

VIBRATION_DATA_CURVED = sorted(DATA_ROOT_PATH.rglob("*data_c*0.csv"))
LANE_DATA_CURVED = DATA_ROOT_PATH / "lane_data_c.csv"

ANSWER_SAMPLE = DATA_ROOT_PATH / "answer_sample.csv"
