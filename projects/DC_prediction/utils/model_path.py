import os
from pathlib import Path

MODEL_PATH_BASE = Path("/opt/railroad/projects/DC_prediction/outputs/baseline-v4")

MODEL_PATH = [
    "baseline-v4-curved-f64-dropout",
    "baseline-v4-curved-f96-dropout",
    ]
MODEL_PATH = [ MODEL_PATH_BASE / Path(path) for path in MODEL_PATH]

TARGET_MODEL_PATH = {
    'curved': MODEL_PATH[0],
    'straight': MODEL_PATH[0]
}