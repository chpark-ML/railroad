import os
from pathlib import Path

MODEL_PATH_BASE = Path("")

MODEL_PATH = [
    "/opt/railroad/projects/DC_prediction/outputs/baseline/baseline-both-epoch50-inter50-LR1e-2",
    # "/opt/railroad/projects/DC_prediction/outputs/baseline-seed-v2/baseline-seed-v2-both-valpre-seed1",
    ]

MODEL_PATH = [ Path(path) for path in MODEL_PATH]

TARGET_MODEL_PATH = {
    'curved': MODEL_PATH[0],
    'straight': MODEL_PATH[0]
}