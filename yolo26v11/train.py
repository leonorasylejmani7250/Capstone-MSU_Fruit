from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--model', default='yolo11n-seg.pt')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    if not (ROOT / 'grape.yaml').is_file():
        p.error('Missing grape.yaml: run prepare_wgisd.py first')
    device = (0 if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}; training: {device}')
    model = YOLO(args.model)
    model.train(data=str(ROOT / 'grape.yaml'), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=device, workers=0, cache=False, amp=device != 'cpu',
                project=str(ROOT / 'runs'), name='yolo11_wgisd', exist_ok=False,
                patience=20, seed=42, plots=True)
    print('Done. Check the latest runs/yolo11_wgisd*/weights/best.pt')

if __name__ == '__main__':
    main()
