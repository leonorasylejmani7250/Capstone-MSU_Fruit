from __future__ import annotations

import argparse

from ultralytics import YOLO

from yolo26_common import ROOT, check_dataset, check_yolo26_version, choose_device, ensure_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate a YOLO segmentation model on WGISD')
    parser.add_argument('--model', required=True, help='Path to best.pt')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--name', default='yolo26_eval')
    args = parser.parse_args()

    check_yolo26_version()
    data = check_dataset()
    model = YOLO(args.model)
    ensure_segmentation(model)
    metrics = model.val(
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=choose_device(args.device), workers=0,
        project=str(ROOT / 'runs'), name=args.name, exist_ok=False, plots=True,
    )
    print(f'Split: {args.split}; model: {args.model}')
    print(f'BOX:  P={metrics.box.mp:.4f}, R={metrics.box.mr:.4f}, '
          f'mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}')
    print(f'MASK: P={metrics.seg.mp:.4f}, R={metrics.seg.mr:.4f}, '
          f'mAP50={metrics.seg.map50:.4f}, mAP50-95={metrics.seg.map:.4f}')
    print('Metrics are for annotated WGISD images, not DJI field videos.')


if __name__ == '__main__':
    main()
