"""Train the model. Default: the pretrained model on the public WGISD grape dataset. With --data and --model: fine-tune
the shipped model on our own labels.

    python train.py                                                                      WGISD → ../data/runs/<pipeline>/wgisd/
    python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard

The run folder holds weights/best.pt, results.csv, results.png and the curves. To ship a run, copy its
weights/best.pt to model/best.pt and update model/README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from pipeline import BASE_MODEL, MODEL, RUNS, WGISD


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', type=Path, default=WGISD, help='dataset YAML (default: WGISD)')
    p.add_argument('--model', default=BASE_MODEL,
                   help=f'starting weights: an Ultralytics name such as {BASE_MODEL} (downloaded) or a file such as model/best.pt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--name', default='wgisd', help=f'run folder {RUNS}/<name>/ (a number is added if it exists)')
    p.add_argument('--device', default='auto', help='auto, cpu, or a CUDA device such as 0')
    args = p.parse_args()
    if not args.data.is_file():
        p.error(f'Missing dataset file {args.data}')
    device = (0 if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'PyTorch {torch.__version__} · device {device} · data {args.data} · starting from {args.model}')
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, workers=0, cache=False, amp=device != 'cpu',
                project=str(RUNS), name=args.name, exist_ok=False, patience=20, seed=42, plots=True)
    best = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    print(f'Best weights: {best}')
    print(f'To ship them: cp {best} {MODEL}  and update model/README.md')


if __name__ == '__main__':
    main()
