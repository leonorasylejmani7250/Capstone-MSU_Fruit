"""Count the grape clusters in a video, or in every video of a folder: an annotated video, a per-frame CSV and the
total in ../data/results/. Default model: model/best.pt.

    python video.py --video ../data/videos/row3.mp4                           one video
    python video.py --video-dir ../data/videos/CodyKresta_1025                 every video in the folder
    python video.py --video ../data/videos/row3.mp4 --max-frames 300           a quick look at the first 300 frames

Every cluster gets an id that follows it from frame to frame (ByteTrack); the total is the number of ids followed
for at least MIN_SECONDS. A whole video adds its total to ../data/results/cluster_counts.csv, the file every
pipeline writes to. --mode detect only draws what the model finds in each frame, without ids or a total.
--imgsz is the size a frame is shrunk to before the model sees it, so it sets how big a cluster looks to the model.
The default, auto, tries every size on five frames of the video and uses the one at which the model is most sure.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline import DATA, MODEL, NAME, RESULTS, load

MIN_SECONDS = 1.0   # an id counts as a cluster once it has been followed for this long; shorter ones are mostly
                    # a cluster the tracker lost and found again, or a leaf mistaken for one


def color_for_id(identifier: int) -> tuple[int, int, int]:
    palette = [(100, 210, 85), (215, 145, 68), (180, 85, 200),
               (76, 185, 230), (215, 110, 165), (96, 200, 175)]
    return palette[int(identifier) % len(palette)]


def clusters(seen: dict[int, int], min_frames: int) -> int:
    """The count so far: ids that have been followed for at least `min_frames` frames."""
    return sum(1 for frames in seen.values() if frames >= min_frames)


def render(frame: np.ndarray, result, seen: dict[int, int], conf_threshold: float,
           mode: str, min_frames: int = 1) -> tuple[np.ndarray, int, int]:
    """Draw the result on the frame. `seen` maps each tracker id to the number of frames it has been followed."""
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
                seen[ident] = seen.get(ident, 0) + 1
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
             f'Clusters so far: {clusters(seen, min_frames)}' if mode == 'track' else 'Mode: detect',
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


VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.mts', '.webm'}
SETTLE_MINUTES = 2      # a file written this recently may still be syncing from SharePoint: skipped until the next run


def find_videos(folder: Path) -> list[Path]:
    """Every video under the folder, any depth, in path order; hidden and partial files and files still being
    written are skipped (a SharePoint sync can land here as it is)."""
    import time
    found, now = [], time.time()
    for f in sorted(folder.rglob('*')):
        rel = f.relative_to(folder)
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS or any(part.startswith(('.', '~$')) for part in rel.parts):
            continue
        if now - f.stat().st_mtime < SETTLE_MINUTES * 60:
            print(f'{rel}: written less than {SETTLE_MINUTES} min ago, still syncing? skipped this time')
            continue
        found.append(f)
    return found


SIZES = (256, 384, 512, 768, 1024, 1280, 1536, 1920, 2560, 3840)   # the input sizes --imgsz auto tries
SURE = 0.25                                                        # a box the model is sure of scores at least this


def sample_frames(video: Path, start_sec: float, max_frames: int, count: int = 5) -> list[np.ndarray]:
    """A few frames spread over the stretch of the video that will be processed."""
    cap = cv2.VideoCapture(str(video))
    first = int(start_sec * (cap.get(cv2.CAP_PROP_FPS) or 0))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(total, first + max_frames) if max_frames else total
    frames = []
    for i in sorted({int(i) for i in np.linspace(first, max(first, last - 1), count)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def sweep(model, frames: list[np.ndarray], conf: float, device) -> list[dict]:
    """The model on the frames at every input size up to their own. One row per size: boxes per frame at `conf`,
    the top score, and `sure`, the summed scores of the boxes that score SURE or more. Only the 60 most confident
    boxes of a frame are looked at: every box costs a full-size mask, and a weak model makes hundreds."""
    full = int(np.ceil(max(max(f.shape[:2]) for f in frames) / 32) * 32)
    rows = []
    for size in [s for s in SIZES if s < full] + [full]:
        scores = [np.zeros(0)]
        for frame in frames:
            boxes = model.predict(source=frame, conf=min(conf, 0.05), imgsz=size, max_det=60, device=device, verbose=False)[0].boxes
            if boxes is not None and len(boxes):
                scores.append(boxes.conf.float().cpu().numpy())
        scores = np.concatenate(scores)
        rows.append({'imgsz': size, 'boxes_per_frame': float((scores >= conf).sum()) / len(frames),
                     'top_score': float(scores.max()) if len(scores) else 0.0,
                     'sure': float(scores[scores >= max(conf, SURE)].sum())})
    return rows


def pick(rows: list[dict], fallback: int) -> int:
    """The size at which the model is most sure. Sure of nothing at any size: the size with the top score.
    Not one box at any size: `fallback`."""
    best = max(rows, key=lambda r: (r['sure'], r['top_score']))
    return best['imgsz'] if best['top_score'] > 0 else fallback


def model_label(weights: Path) -> str:
    """How a weights file is named in the count table: model/best.pt → the pipeline's name; a training run's
    weights/best.pt → the run's name."""
    weights = Path(weights).resolve()
    if weights == MODEL.resolve():
        return NAME
    return weights.parents[1].name if weights.name == 'best.pt' else weights.stem


COUNTS = DATA / 'results' / 'cluster_counts.csv'      # one line per counted video and model; every pipeline adds to it


def video_key(video: Path) -> str:
    """How a video is named in the count table: its path under ../data/videos/, else its file name."""
    videos = (DATA / 'videos').resolve()
    return video.resolve().relative_to(videos).as_posix() if video.resolve().is_relative_to(videos) else video.name


def record_count(video: Path, model: str, total: int, seconds: float) -> Path:
    """Add a video's total to the count table."""
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    new = not COUNTS.exists() or not COUNTS.stat().st_size
    with COUNTS.open('a', newline='', encoding='utf-8') as f:
        if new:
            csv.writer(f).writerow(['video', 'model', 'clusters', 'seconds', 'pipeline', 'written'])
        csv.writer(f).writerow([video_key(video), model, total, round(seconds, 1), NAME, time.strftime('%Y-%m-%d %H:%M')])
    return COUNTS


def counted(model: str) -> set[str]:
    """The videos this model has a total for already."""
    if not COUNTS.exists():
        return set()
    with COUNTS.open(newline='', encoding='utf-8') as f:
        return {row['video'] for row in csv.DictReader(f) if row['model'] == model}


def process(video: Path, model, args, output: Path) -> None:
    """One video → annotated video at `output`, a per-frame CSV next to it, and with ids the total."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {video}')
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
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Unable to create {output}')
    seen: dict[int, int] = {}
    min_frames = max(1, round(MIN_SECONDS * fps))
    csv_path = output.with_suffix('.csv')
    processed = found = 0
    device = 0 if torch.cuda.is_available() else 'cpu'
    size = args.imgsz
    if size == 'auto':
        frames = sample_frames(video, args.start_sec, args.max_frames)
        rows = sweep(model, frames, args.conf, device) if frames else []           # an unreadable video has no frames
        size = pick(rows, args.trained_size or 768) if rows else args.trained_size or 768
        print('imgsz  boxes per frame  top score')
        for r in rows:
            print(f"{r['imgsz']:5}  {r['boxes_per_frame']:15.1f}  {r['top_score']:9.2f}" + ('   <- used' if r['imgsz'] == size else ''))
    try:
        with csv_path.open('w', newline='', encoding='utf-8') as file:
            log = csv.writer(file)
            log.writerow(['frame', 'visible_now', 'tracked_now', 'clusters_so_far'])
            while args.max_frames == 0 or processed < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                kwargs = dict(source=frame, conf=args.conf, imgsz=size,
                              device=device, verbose=False, retina_masks=True)
                if args.mode == 'track':
                    result = model.track(**kwargs, persist=processed > 0, tracker=args.tracker)[0]      # ids start over with each video
                else:
                    result = model.predict(**kwargs)[0]
                rendered, visible, tracked = render(frame, result, seen, args.conf, args.mode, min_frames)
                writer.write(rendered)
                processed += 1
                found += visible
                log.writerow([processed, visible, tracked, clusters(seen, min_frames)])
                if processed % 300 == 0:
                    print(f'Frame {processed} | visible={visible}, clusters so far={clusters(seen, min_frames)}', flush=True)
    finally:
        writer.release()
        cap.release()
    print(f'Saved {processed} frames: {output}')
    print(f'Per-frame CSV: {csv_path}')
    if args.mode == 'track':
        total = clusters(seen, min_frames)
        print(f'Clusters counted: {total} (ids followed for at least {MIN_SECONDS:g} s)')
        if not args.start_sec and not args.max_frames:          # a whole video: its total goes into the shared table
            print(f'Total written to {record_count(video, model_label(args.model), total, processed / fps)}')
    if processed and not found:
        print('No clusters found. ' + ('The model is sure of nothing in this video at any input size: the footage is unlike the '
                                       'photos it was trained on. Label clips of it and fine-tune (../LABELLING.md).'
                                       if args.imgsz == 'auto' else 'Leave --imgsz out: the default tries every input size.'), flush=True)


def main():
    p = argparse.ArgumentParser(description=' '.join(__doc__.split('\n\n')[0].split()))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=Path, help='one video')
    src.add_argument('--video-dir', type=Path, help='every video under a folder, any depth; done ones are skipped')
    p.add_argument('--model', type=Path, default=MODEL, help='weights (default: model/best.pt)')
    p.add_argument('--output', type=Path, default=None,
                   help='annotated video, or with --video-dir the output folder (default: ../data/results/<pipeline>/[<folder name>/]<video name>_<mode>.mp4)')
    p.add_argument('--mode', choices=['track', 'detect'], default='track',
                   help='track: ids across frames and a total (default); detect: only what is in each frame')
    p.add_argument('--conf', type=float, default=0.30)
    p.add_argument('--imgsz', type=lambda v: v if v == 'auto' else int(v), default='auto',
                   help='input size: auto (the size at which the model is most sure on this video) or a number')
    p.add_argument('--max-frames', type=int, default=0, help='with --video: frames to process (default 0: the whole video)')
    p.add_argument('--start-sec', type=float, default=0., help='with --video: where to start, in seconds')
    p.add_argument('--tracker', default='bytetrack.yaml')
    p.add_argument('--overwrite', action='store_true', help='with --video-dir: redo videos that already have an output')
    args = p.parse_args()
    if not 0 < args.conf <= 1 or (args.imgsz != 'auto' and args.imgsz < 160):
        p.error('Require confidence in (0,1] and imgsz >=160')
    if not args.model.is_file():
        p.error(f'Missing weights: {args.model}')
    if args.start_sec < 0 or args.max_frames < 0:
        p.error('start-sec and max-frames must be nonnegative')
    model, args.trained_size = load(args.model)

    if args.video:
        if not args.video.is_file():
            p.error(f'Missing video: {args.video}')
        output = args.output or RESULTS / f'{args.video.stem}_{args.mode}.mp4'
        process(args.video, model, args, output)
    else:
        if not args.video_dir.is_dir():
            p.error(f'Missing folder: {args.video_dir}')
        videos = find_videos(args.video_dir)
        if not videos:
            p.error(f'No videos under {args.video_dir}')
        out_dir = args.output or RESULTS / args.video_dir.name
        args.max_frames, args.start_sec = 0, 0.
        done = counted(model_label(args.model)) if args.mode == 'track' else set()
        failed = []
        for i, video in enumerate(videos, 1):
            rel = video.relative_to(args.video_dir)                      # subfolders are kept in the output
            output = out_dir / rel.parent / f'{video.stem}_{args.mode}.mp4'
            finished = video_key(video) in done if args.mode == 'track' else output.with_suffix('.csv').exists()
            if finished and not args.overwrite:
                print(f'[{i}/{len(videos)}] {rel}: done already, skipping (--overwrite to redo)')
                continue
            print(f'[{i}/{len(videos)}] {rel}')
            try:
                process(video, model, args, output)
            except RuntimeError as e:                                     # one unreadable file does not stop the folder
                print(f'  skipped: {e}')
                failed.append(str(rel))
        if failed:
            print(f'{len(failed)} video(s) could not be read: ' + ', '.join(failed))
    print('A count is the number of ids the tracker followed, not a hand count, and not a yield estimate.')


if __name__ == '__main__':
    main()
