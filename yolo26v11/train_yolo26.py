from __future__ import annotations

import argparse

from ultralytics import YOLO

from yolo26_common import (
    ROOT, check_dataset, check_yolo26_version, choose_device, ensure_segmentation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train YOLO26 instance segmentation on WGISD')
    parser.add_argument('--model', default='yolo26n-seg.pt', help='Pretrained segmentation checkpoint')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index (e.g. 0)')
    parser.add_argument('--name', default='yolo26_wgisd', help='Output run name')
    parser.add_argument('--workers', type=int, default=0, help='Windows-friendly default')
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 160 or args.batch < 1 or args.workers < 0:
        parser.error('epochs, batch must be positive; imgsz >= 160; workers >= 0')
    check_yolo26_version()
    data = check_dataset()
    device = choose_device(args.device)
    model = YOLO(args.model)
    ensure_segmentation(model)
    print(f'Model: {args.model}; task: {model.task}; device: {device}', flush=True)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, workers=args.workers, cache=False, amp=device != 'cpu',
        project=str(ROOT / 'runs'), name=args.name, exist_ok=False,
        patience=args.patience, seed=42, plots=True,
    )
    # Ultralytics auto-increments the run name if it already exists.
    print('Training complete. Check runs/yolo26_wgisd*/weights/best.pt', flush=True)


if __name__ == '__main__':
    main()
