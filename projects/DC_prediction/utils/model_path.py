import os
from pathlib import Path

MODEL_PATH_BASE = Path("/opt/railroad/projects/DC_prediction/outputs/alpha-test")

MODEL_PATH = [
    "alpha-test-curved-f64-alpha1.0-linear",
    "alpha-test-straight-f64-alpha1.0-linear",
    ]
MODEL_PATH = [ MODEL_PATH_BASE / Path(path) for path in MODEL_PATH]

TARGET_MODEL_PATH = {
    'curved': MODEL_PATH[0],
    'straight': MODEL_PATH[1]
}