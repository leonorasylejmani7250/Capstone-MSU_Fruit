"""Run a trained YOLO11 checkpoint on a video: an annotated output video (detection or ByteTrack tracking) plus a per-frame CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def color_for_id(identifier: int) -> tuple[int, int, int]:
    palette = [(100, 210, 85), (215, 145, 68), (180, 85, 200),
               (76, 185, 230), (215, 110, 165), (96, 200, 175)]
    return palette[int(identifier) % len(palette)]


def render(frame: np.ndarray, result, seen_ids: set[int], conf_threshold: float,
           mode: str) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    out = frame.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        n = 0
        track_now = 0
        confidences = []
    else:
        xyxy = boxes.xyxy.int().cpu().numpy()
        confidences = boxes.conf.float().cpu().numpy().tolist()
        # Some tracker versions return detections but no track IDs.
        ids = (boxes.id.int().cpu().numpy().tolist()
               if getattr(boxes, 'id', None) is not None else [])
        polygons = result.masks.xy if result.masks is not None else []
        n = len(xyxy)
        track_now = min(len(ids), n)
        overlay = out.copy()
        for i, (bb, conf) in enumerate(zip(xyxy, confidences)):
            ident = int(ids[i]) if i < len(ids) else None
            color = color_for_id(ident if ident is not None else i)
            if i < len(polygons) and len(polygons[i]) >= 3:
                pts = np.asarray(polygons[i], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, .32, out, .68, 0)
        for i, (bb, conf) in enumerate(zip(xyxy, confidences)):
            ident = int(ids[i]) if i < len(ids) else None
            if ident is not None:
                seen_ids.add(ident)
            color = color_for_id(ident if ident is not None else i)
            x1, y1, x2, y2 = map(int, bb)
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if i < len(polygons) and len(polygons[i]) >= 3:
                cv2.polylines(out, [np.asarray(polygons[i], dtype=np.int32)], True, color, 2)
            # Object ID on upper left; confidence on upper right of each box.
            text_id = f'ID {ident}' if ident is not None else 'Grape'
            text_conf = f'{conf * 100:.0f}%'
            ty = max(19, y1 - 7)
            cv2.putText(out, text_id, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, text_id, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, color, 2, cv2.LINE_AA)
            tw = cv2.getTextSize(text_conf, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)[0][0]
            conf_x = max(0, x2 - tw)
            cv2.putText(out, text_conf, (conf_x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, text_conf, (conf_x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, color, 2, cv2.LINE_AA)

    lines = [f'Grapes visible: {n}', f'Tracked now: {track_now}',
             f'Track IDs seen: {len(seen_ids)}' if mode == 'track' else 'Mode: detect',
             f'Best confidence: {max(confidences) * 100:.0f}%' if confidences else 'Best confidence: --',
             f'Conf threshold: {conf_threshold * 100:.0f}%']
    scale, thick, gap, margin = .55, 2, 25, 14
    max_text = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
                   for line in lines)
    panel_w = min(w, max_text + 2 * margin)
    panel_h = min(h, (len(lines) + 1) * gap)
    px = max(0, w - panel_w - 8)
    py = 8
    cv2.rectangle(out, (px, py), (min(w-1, px+panel_w), min(h-1, py+panel_h)), (24, 24, 24), -1)
    for j, line in enumerate(lines):
        text_y = py + (j + 1) * gap
        if text_y < h:
            cv2.putText(out, line, (px + margin, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (248, 248, 248), thick, cv2.LINE_AA)
    return out, n, track_now


def main():
    p = argparse.ArgumentParser(description='Segment grape clusters in video')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'grapes.mp4')
    p.add_argument('--mode', choices=['detect', 'track'], default='detect')
    p.add_argument('--conf', type=float, default=0.30)
    p.add_argument('--imgsz', type=int, default=768)
    p.add_argument('--max-frames', type=int, default=300, help='0 means entire video')
    p.add_argument('--start-sec', type=float, default=0.)
    p.add_argument('--tracker', default='bytetrack.yaml')
    args = p.parse_args()
    if not 0 < args.conf <= 1 or args.imgsz < 160:
        p.error('Require confidence in (0,1] and imgsz >=160')
    if not args.video.is_file():
        p.error(f'Missing video: {args.video}')
    if not args.model.is_file():
        p.error(f'Missing trained model: {args.model}')
    if args.start_sec < 0 or args.max_frames < 0:
        p.error('start-sec and max-frames must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError('Video has invalid frame rate')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w < 1 or h < 1:
        cap.release()
        raise RuntimeError('Video has invalid resolution')
    if args.start_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Unable to create {args.output}')
    seen_ids: set[int] = set()
    csv_path = args.output.with_suffix('.csv')
    processed = 0
    device = 0 if torch.cuda.is_available() else 'cpu'
    try:
        with csv_path.open('w', newline='', encoding='utf-8') as file:
            log = csv.writer(file)
            log.writerow(['frame', 'visible_now', 'tracked_now', 'tracker_ids_seen'])
            while args.max_frames == 0 or processed < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                kwargs = dict(source=frame, conf=args.conf, imgsz=args.imgsz,
                              device=device, verbose=False, retina_masks=True)
                if args.mode == 'track':
                    result = model.track(**kwargs, persist=True, tracker=args.tracker)[0]
                else:
                    result = model.predict(**kwargs)[0]
                rendered, visible, tracked = render(frame, result, seen_ids, args.conf, args.mode)
                writer.write(rendered)
                processed += 1
                log.writerow([processed, visible, tracked, len(seen_ids)])
                if processed % 30 == 0:
                    print(f'Frame {processed} | visible={visible}, tracked={tracked}, IDs seen={len(seen_ids)}', flush=True)
    finally:
        writer.release()
        cap.release()
    print(f'Saved {processed} frames: {args.output}')
    print(f'Diagnostics CSV: {csv_path}')
    print('Track IDs seen != verified unique clusters; no yield estimate is claimed.')

if __name__ == '__main__':
    main()
