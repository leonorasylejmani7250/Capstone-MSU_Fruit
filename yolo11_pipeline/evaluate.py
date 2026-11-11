"""Box and mask mAP on the WGISD test photos. Default: the shipped model. Several weights make a table.

    python evaluate.py                                                   model/best.pt on the 27 test photos
    python evaluate.py --model ../data/runs/<pipeline>/vineyard/weights/best.pt --split val
    python evaluate.py --model model/best.pt <other weights> ...         → ../data/results/<pipeline>/evaluate_<split>.csv

Plots (PR curve, confusion matrix) go to ../data/runs/<pipeline>/eval_<split>_<name>/ when one model is scored.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from pipeline import MODEL, RESULTS, RUNS, WGISD, load


def main():
    p = argparse.ArgumentParser(description=' '.join(__doc__.split('\n\n')[0].split()))
    p.add_argument('--model', type=Path, nargs='+', default=[MODEL], help='weights to score (default: model/best.pt)')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--imgsz', type=int, default=None, help='image size (default: the size each model was trained at)')
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    if not WGISD.is_file():
        p.error(f'Missing {WGISD}: python ../data/wgisd_yolo/prepare_wgisd.py')
    for m in args.model:
        if not m.is_file():
            p.error(f'Missing weights: {m}')
    device = 0 if torch.cuda.is_available() else 'cpu'
    rows = []
    for m in args.model:
        tag = m.parents[1].name if m.parent.name == 'weights' else m.parent.name      # runs/<run>/weights/best.pt → <run>
        model, trained_size = load(m)
        imgsz = args.imgsz or trained_size or 512                 # WGISD photos are scored at the size the model learned them at
        metrics = model.val(data=str(WGISD), split=args.split, imgsz=imgsz, batch=args.batch,
                            device=device, workers=0, project=str(RUNS), name=f'eval_{args.split}_{tag}',
                            exist_ok=True, plots=len(args.model) == 1)
        speed = getattr(metrics, 'speed', {}) or {}
        rows.append({'model': str(m), 'split': args.split, 'imgsz': imgsz,
                     'box_mAP50': round(metrics.box.map50, 4), 'box_mAP50-95': round(metrics.box.map, 4),
                     'mask_mAP50': round(metrics.seg.map50, 4), 'mask_mAP50-95': round(metrics.seg.map, 4),
                     'mask_precision': round(metrics.seg.mp, 4), 'mask_recall': round(metrics.seg.mr, 4),
                     'ms_per_image': round(float(speed.get('inference', 0.0)), 1)})
        print(f'{m}\n  BOX  mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}')
        print(f'  MASK mAP50={metrics.seg.map50:.4f}, mAP50-95={metrics.seg.map:.4f}')
    if len(rows) > 1:
        out = RESULTS / f'evaluate_{args.split}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'Table: {out}')
    print(f'Scores on WGISD {args.split} photos. The leaderboard on our own clips is ../sam2_pipeline: python fruit.py evaluate')


if __name__ == '__main__':
    main()
