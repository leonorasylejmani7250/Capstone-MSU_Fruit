"""Compare YOLO11-seg and YOLO26-seg on the SAME annotated WGISD split."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from yolo26_common import ROOT, check_dataset, check_yolo26_version, choose_device, ensure_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description='Segmentation benchmark on WGISD')
    parser.add_argument('--yolo11', type=Path, default=ROOT / 'runs' / 'yolo11_wgisd2' / 'weights' / 'best.pt')
    parser.add_argument('--yolo26', type=Path, default=ROOT / 'runs' / 'yolo26_wgisd' / 'weights' / 'best.pt')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'yolo26' / 'comparison.csv')
    args = parser.parse_args()
    check_yolo26_version()
    data = check_dataset()
    rows = []
    for label, checkpoint in (('YOLO11n-seg', args.yolo11), ('YOLO26n-seg', args.yolo26)):
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {checkpoint}')
        model = YOLO(str(checkpoint))
        ensure_segmentation(model)
        metrics = model.val(
            data=str(data), split=args.split, imgsz=args.imgsz,
            batch=args.batch, device=choose_device(args.device), workers=0,
            project=str(ROOT / 'runs'), name=f'compare_{label.lower()}',
            exist_ok=False, plots=False,
        )
        speed = getattr(metrics, 'speed', {}) or {}
        row = {
            'model': label, 'split': args.split, 'imagesz': args.imgsz,
            'mask_precision': round(metrics.seg.mp, 5),
            'mask_recall': round(metrics.seg.mr, 5),
            'mask_map50': round(metrics.seg.map50, 5),
            'mask_map50_95': round(metrics.seg.map, 5),
            'box_map50': round(metrics.box.map50, 5),
            'box_map50_95': round(metrics.box.map, 5),
            'inference_ms_per_image': round(float(speed.get('inference', 0.0)), 3),
        }
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Comparison saved: {args.output}')
    print('Evaluate on val while tuning; reserve test for final reporting.')


if __name__ == '__main__':
    main()
