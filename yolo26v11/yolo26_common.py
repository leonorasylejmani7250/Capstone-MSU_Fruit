"""Shared safeguards for the WGISD grape segmentation YOLO26 experiment."""
from __future__ import annotations

from pathlib import Path
from importlib.metadata import version

import yaml
from packaging.version import Version

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / 'data' / 'wgisd_seg'
DATASET_CONFIG = ROOT / 'grape.yaml'


def check_yolo26_version() -> None:
    installed = version('ultralytics')
    if Version(installed) < Version('8.4.0'):
        raise RuntimeError(
            f'Ultralytics {installed} does not support YOLO26. '
            'Use the .venv26 Python interpreter and install requirements-yolo26.txt.'
        )


def check_dataset() -> Path:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError('Missing grape.yaml; run prepare_wgisd.py first.')
    with DATASET_CONFIG.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or 'path' not in config:
        raise ValueError('Invalid grape.yaml: missing path.')
    dataset_root = Path(str(config['path'])).resolve()
    if dataset_root != DATASET_DIR.resolve():
        raise ValueError(
            f'grape.yaml points to {dataset_root}, but this project expects '
            f'{DATASET_DIR.resolve()}. Run prepare_wgisd.py again to fix paths.'
        )
    for split in ('train', 'val', 'test'):
        images = DATASET_DIR / 'images' / split
        labels = DATASET_DIR / 'labels' / split
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(f'Missing WGISD {split} images or labels; run prepare_wgisd.py.')
    return DATASET_CONFIG


def choose_device(requested: str) -> str | int:
    import torch
    if requested.lower() != 'auto':
        return int(requested) if requested.isdigit() else requested
    return 0 if torch.cuda.is_available() else 'cpu'


def ensure_segmentation(model) -> None:
    if model.task != 'segment':
        raise ValueError(
            f'Expected an instance-segmentation checkpoint, got task={model.task!r}. '
            'Use yolo26n-seg.pt, not yolo26n.pt.'
        )
