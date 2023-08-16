from enum import Enum


class RunMode(Enum):
    TRAIN = 'train'
    VALIDATE = 'val'
    TEST = 'test'
    VALIDATE_ANALYSIS = 'val_analysis'
    TEST_ANALYSIS = 'test_analysis'