"""Evaluate a YOLO segmentation checkpoint (default: model/best.pt) on the WGISD test split and print box and mask mAP.

Ultralytics' evaluation plots and curves go to data/runs/yolo26/eval_<split>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent       # yolo26_pipeline/
DATA = ROOT.parent / 'data'                  # the shared data folder of the repository
MODEL = ROOT / 'model' / 'best.pt'           # the model this pipeline ships


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', type=Path, default=MODEL, help='weights to score (default: model/best.pt)')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    if not (ROOT / 'grape.yaml').is_file():
        p.error('Missing grape.yaml. Run prepare_wgisd.py')
    if not args.model.is_file():
        p.error(f'Missing weights: {args.model} (train.py makes them; model/README.md says what goes in model/)')
    metrics = YOLO(str(args.model)).val(data=str(ROOT / 'grape.yaml'), split=args.split,
                                        imgsz=args.imgsz, batch=args.batch,
                                        device=0 if torch.cuda.is_available() else 'cpu', workers=0,
                                        project=str(DATA / 'runs' / 'yolo26'), name=f'eval_{args.split}',
                                        exist_ok=False)
    print(f'BOX  mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}')
    print(f'MASK mAP50={metrics.seg.map50:.4f}, mAP50-95={metrics.seg.map:.4f}')
    print(f'These scores are on WGISD {args.split} images, not on vineyard video.')


if __name__ == '__main__':
    main()
