"""Train YOLO26n-seg on the WGISD grape clusters (grape.yaml). The run is saved under data/runs/yolo26/<name>/.

When a run is good enough to ship, copy its weights/best.pt to model/best.pt (see model/README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent       # yolo26_pipeline/
DATA = ROOT.parent / 'data'                  # the shared data folder of the repository


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--model', default='yolo26n-seg.pt', help='base weights; Ultralytics downloads them on first use')
    p.add_argument('--name', default='wgisd', help='run name under data/runs/yolo26/ (a number is added if it exists)')
    p.add_argument('--device', default='auto', help="auto, cpu, or a CUDA device such as 0")
    args = p.parse_args()
    if not (ROOT / 'grape.yaml').is_file():
        p.error('Missing grape.yaml: run prepare_wgisd.py first')
    device = (0 if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}; training on: {device}')
    model = YOLO(args.model)
    model.train(data=str(ROOT / 'grape.yaml'), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=device, workers=0, cache=False, amp=device != 'cpu',
                project=str(DATA / 'runs' / 'yolo26'), name=args.name, exist_ok=False,
                patience=20, seed=42, plots=True)
    best = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    print(f'Done. Best weights: {best}')
    print(f'To ship them: copy that file to {ROOT / "model" / "best.pt"} and update model/README.md')


if __name__ == '__main__':
    main()
