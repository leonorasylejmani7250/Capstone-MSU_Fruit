from __future__ import annotations
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    if not (ROOT / 'grape.yaml').is_file():
        p.error('Missing grape.yaml. Run prepare_wgisd.py')
    metrics = YOLO(str(args.model)).val(data=str(ROOT / 'grape.yaml'), split='test',
               imgsz=args.imgsz, batch=args.batch, device=0 if torch.cuda.is_available() else 'cpu',
               workers=0, project=str(ROOT / 'runs'), name='eval_test', exist_ok=False)
    print(f'BOX  mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}')
    print(f'MASK mAP50={metrics.seg.map50:.4f}, mAP50-95={metrics.seg.map:.4f}')
    print('These scores evaluate WGISD test images, not unlabeled DJI videos.')

if __name__ == '__main__':
    main()
