"""The names and folders of this pipeline. Every script in this folder imports them."""
from pathlib import Path

NAME = 'yolo11'                                     # this pipeline; names its folders under ../data/runs and ../data/results
BASE_MODEL = 'yolo11n-seg.pt'                       # Ultralytics' pretrained weights that train.py starts from
ROOT = Path(__file__).resolve().parent              # this folder
DATA = ROOT.parent / 'data'                         # the shared data folder of the repository
MODEL = ROOT / 'model' / 'best.pt'                  # the model this pipeline ships
WGISD = DATA / 'wgisd_yolo' / 'grape.yaml'          # the public WGISD grape dataset as YOLO labels (../data/README.md)
RUNS = DATA / 'runs' / NAME                         # training runs: <run>/weights/best.pt, results.csv, curves
RESULTS = DATA / 'results' / NAME                   # everything else this pipeline writes
