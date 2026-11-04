"""Train the pipeline's YOLO model (YOLO11n-seg here) on the WGISD grape clusters (grape.yaml), or fine-tune it on other labels (--data). Runs go to data/runs/<pipeline>/<name>/.

Fine-tune the shipped model on our own vineyard labels (exported by the SAM 2.1 pipeline, `fruit.py export-yolo`):
    python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --name vineyard
When a run is good enough to ship, copy its weights/best.pt to model/best.pt (see model/README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent       # this pipeline's folder
DATA = ROOT.parent / 'data'                  # the shared data folder of the repository
PIPELINE = ROOT.name.replace('_pipeline', '') # yolo26 or yolo11: names this pipeline's folders under data/runs and data/results
BASE_MODEL = {'yolo26': 'yolo26n-seg.pt', 'yolo11': 'yolo11n-seg.pt'}.get(PIPELINE, 'yolo26n-seg.pt')   # Ultralytics' pretrained weights to start from


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--data', type=Path, default=ROOT / 'grape.yaml', help='dataset YAML (default: grape.yaml, WGISD)')
    p.add_argument('--model', default=BASE_MODEL,
                   help='starting weights: a bare Ultralytics name (downloaded) or a file such as model/best.pt')
    p.add_argument('--name', default='wgisd', help=f'run name under data/runs/{PIPELINE}/ (a number is added if it exists)')
    p.add_argument('--device', default='auto', help="auto, cpu, or a CUDA device such as 0")
    args = p.parse_args()
    if not args.data.is_file():
        p.error(f'Missing dataset file {args.data}' + (': run prepare_wgisd.py first' if args.data.name == 'grape.yaml' else ''))
    device = (0 if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}; training on: {device}; data: {args.data}')
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=device, workers=0, cache=False, amp=device != 'cpu',
                project=str(DATA / 'runs' / PIPELINE), name=args.name, exist_ok=False,
                patience=20, seed=42, plots=True)
    best = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    print(f'Done. Best weights: {best}')
    print(f'To ship them: copy that file to {ROOT / "model" / "best.pt"} and update model/README.md')


if __name__ == '__main__':
    main()
