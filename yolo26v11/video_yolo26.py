from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from video import render  # Reuse the known-working YOLO11 drawing logic.
from yolo26_common import ROOT, check_yolo26_version, ensure_segmentation


def main() -> None:
    parser = argparse.ArgumentParser(description='YOLO26 grape-cluster video segmentation')
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--model', type=Path, required=True, help='Trained YOLO26 best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'yolo26_video.mp4')
    parser.add_argument('--mode', choices=['detect', 'track'], default='detect')
    parser.add_argument('--conf', type=float, default=0.30)
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--max-frames', type=int, default=300, help='0 = process whole video')
    parser.add_argument('--start-sec', type=float, default=0.0)
    parser.add_argument('--tracker', default='bytetrack.yaml')
    parser.add_argument('--nms-free', action='store_true', help='YOLO26 one-to-one head (experimental)')
    args = parser.parse_args()

    if not args.video.is_file() or not args.model.is_file():
        parser.error('The video and trained model files must exist.')
    if not 0 < args.conf <= 1 or args.imgsz < 160:
        parser.error('Confidence must be in (0, 1] and imgsz >= 160.')
    if args.start_sec < 0 or args.max_frames < 0:
        parser.error('start-sec and max-frames cannot be negative.')
    if args.nms_free and args.mode == 'track':
        parser.error('--nms-free is supported here only in --mode detect.')

    check_yolo26_version()
    model = YOLO(str(args.model))
    ensure_segmentation(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or w < 1 or h < 1:
        cap.release()
        raise RuntimeError('Invalid input video FPS or dimensions.')
    if args.start_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open output video for writing: {args.output}')
    csv_path = args.output.with_suffix('.csv')
    device = 0 if torch.cuda.is_available() else 'cpu'
    seen_ids: set[int] = set()
    count = 0
    try:
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            log = csv.writer(handle)
            log.writerow(['frame', 'visible_now', 'tracked_now', 'tracker_ids_seen'])
            while args.max_frames == 0 or count < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                kwargs = dict(
                    source=frame, conf=args.conf, imgsz=args.imgsz,
                    device=device, verbose=False, retina_masks=True,
                )
                if args.mode == 'track':
                    result = model.track(**kwargs, persist=True, tracker=args.tracker)[0]
                else:
                    if args.nms_free:
                        kwargs['nms'] = False
                    result = model.predict(**kwargs)[0]
                annotated, visible, tracked = render(frame, result, seen_ids, args.conf, args.mode)
                writer.write(annotated)
                count += 1
                log.writerow([count, visible, tracked, len(seen_ids)])
                if count % 30 == 0:
                    print(f'Frame {count} | visible={visible} tracked={tracked} IDs={len(seen_ids)}', flush=True)
    finally:
        cap.release()
        writer.release()
    print(f'Saved {count} frames: {args.output}')
    print(f'CSV: {csv_path}')
    print('Tracked IDs are NOT a verified unique-cluster count or fruit yield estimate.')


if __name__ == '__main__':
    main()
