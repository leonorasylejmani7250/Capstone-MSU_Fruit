"""Evaluate one or more YOLO segmentation checkpoints on a WGISD split (default: model/best.pt on test) and print box and mask mAP.

    python evaluate.py                                                          the shipped model, test split
    python evaluate.py --model model/best.pt ../data/results/yolo11/best.pt     two models side by side → data/results/yolo26/evaluate_test.csv

Ultralytics' plots and curves go to data/runs/yolo26/eval_<split>_<model>/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent       # yolo26_pipeline/
DATA = ROOT.parent / 'data'                  # the shared data folder of the repository
MODEL = ROOT / 'model' / 'best.pt'           # the model this pipeline ships


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', type=Path, nargs='+', default=[MODEL], help='weights to score (default: model/best.pt)')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--imgsz', type=int, default=512, help='the size the models were trained at')
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    if not (ROOT / 'grape.yaml').is_file():
        p.error('Missing grape.yaml. Run prepare_wgisd.py')
    for m in args.model:
        if not m.is_file():
            p.error(f'Missing weights: {m} (train.py makes them; model/README.md says what goes in model/)')
    device = 0 if torch.cuda.is_available() else 'cpu'
    rows = []
    for m in args.model:
        tag = m.parents[1].name if m.parent.name == 'weights' else m.parent.name       # runs/<run>/weights/best.pt → <run>
        metrics = YOLO(str(m)).val(data=str(ROOT / 'grape.yaml'), split=args.split, imgsz=args.imgsz,
                                   batch=args.batch, device=device, workers=0,
                                   project=str(DATA / 'runs' / 'yolo26'), name=f'eval_{args.split}_{tag}',
                                   exist_ok=True, plots=len(args.model) == 1)
        speed = getattr(metrics, 'speed', {}) or {}
        rows.append({'model': str(m), 'split': args.split, 'imgsz': args.imgsz,
                     'box_mAP50': round(metrics.box.map50, 4), 'box_mAP50-95': round(metrics.box.map, 4),
                     'mask_mAP50': round(metrics.seg.map50, 4), 'mask_mAP50-95': round(metrics.seg.map, 4),
                     'mask_precision': round(metrics.seg.mp, 4), 'mask_recall': round(metrics.seg.mr, 4),
                     'ms_per_image': round(float(speed.get('inference', 0.0)), 1)})
        print(f'{m}\n  BOX  mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}')
        print(f'  MASK mAP50={metrics.seg.map50:.4f}, mAP50-95={metrics.seg.map:.4f}')
    if len(rows) > 1:
        out = DATA / 'results' / 'yolo26' / f'evaluate_{args.split}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'Table: {out}')
    print(f'Scores on WGISD {args.split} images, not on vineyard video.')


if __name__ == '__main__':
    main()
