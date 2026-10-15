"""Filesystem locations for the publication package."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_path(relative):
    return str(Path(os.environ.get('BRAIN_NETS_DATA_DIR', ROOT / 'Datasets')) / relative)


def output_path(relative):
    path = Path(os.environ.get('BRAIN_NETS_OUTPUT_DIR', ROOT / 'outputs')) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
