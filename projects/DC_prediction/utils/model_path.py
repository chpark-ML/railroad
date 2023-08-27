import os
from pathlib import Path

MODEL_PATH_BASE = Path("/opt/railroad/projects/DC_prediction/outputs/baseline")

# MODEL_PATH = [
#     "baseline-epoch30-both"]

MODEL_PATH = [
    "baseline-both-epoch50-inter50-LR1e-2",
    ]
MODEL_PATH = [ MODEL_PATH_BASE / Path(path) for path in MODEL_PATH]

TARGET_MODEL_PATH = {
    'curved': MODEL_PATH[0],
    'straight': MODEL_PATH[0]
}