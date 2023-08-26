import os
from pathlib import Path

MODEL_PATH_BASE = Path("/opt/railroad/projects/DC_prediction/outputs")

MODEL_PATH = [
    "baseline-uniform/baseline-uniform-curved-f32-LR1e-2-mode-uniform-max1.0/2023-08-26_14-50-51",
    "baseline-uniform/baseline-uniform-curved-f32-LR1e-2-mode-uniform-max4.0/2023-08-26_14-50-59",
    "baseline-uniform/baseline-uniform-curved-f32-LR1e-2-mode-uniform-max16.0/2023-08-26_14-51-14",
    "baseline-uniform/baseline-uniform-curved-f32-LR1e-2-mode-uniform-max64.0/2023-08-26_14-51-36",
    ]
MODEL_PATH = [ MODEL_PATH_BASE / Path(path) for path in MODEL_PATH]

TARGET_MODEL_PATH = {
    'curved': MODEL_PATH[0],
    'straight': MODEL_PATH[0]
}