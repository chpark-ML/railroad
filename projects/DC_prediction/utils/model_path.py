import os
from pathlib import Path

MODEL_PATH_BASE = Path("/opt/railroad/model")

MODEL_PATH = [
    "best",
    ]

MODEL_PATH = [ MODEL_PATH_BASE / Path(path) for path in MODEL_PATH]

TARGET_MODEL_PATH = {
    'curved': MODEL_PATH[0],
    'straight': MODEL_PATH[0]
}