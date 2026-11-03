"""Run SAM 3.1 with a text prompt ("grape cluster") on our labelled clips or on a video, and write its masks to data/.

Two jobs:
  python run_sam3.py clips                     masks for the held-out val clips → data/predictions/sam3_text/<clip>/<frame>.png
                                               (then `python ../sam2_pipeline/fruit.py evaluate` scores them with every other model)
  python run_sam3.py video ../data/videos/row3.mp4 --seconds 30   an overlay video and per-frame counts → data/results/sam3/

Needs Meta's `sam3` package, the gated SAM 3.1 weights and a CUDA GPU (README.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent       # sam3_pipeline/
# The project folder: the repository's data/ unless $PROJECT_DIR (sam2_pipeline/hpcc/config.sh) or --data says otherwise.
DATA = Path(os.environ.get('PROJECT_DIR') or ROOT.parent / 'data').expanduser()
MODEL_DIR = ROOT / 'model'                   # where the downloaded SAM 3.1 checkpoint lives (model/README.md)
PALETTE = [(100, 210, 85), (215, 145, 68), (180, 85, 200), (76, 185, 230), (215, 110, 165), (96, 200, 175)]


def build_predictor(version: str, max_num_objects: int, checkpoint: Path | None):
    """Meta's video predictor. SAM 3.1 adds Object Multiplex (many objects per clip); SAM 3 is the original."""
    import sam3.model_builder as mb
    args = {}
    if checkpoint:
        args['checkpoint_path'] = str(checkpoint)
    if version == 'sam3.1':
        return mb.build_sam3_multiplex_video_predictor(max_num_objects=max_num_objects, **args)
    return mb.build_sam3_video_predictor(**args)


def track_text(predictor, resource: Path, text: str, hw: tuple[int, int], n_frames: int,
               prob_thresh: float) -> np.ndarray:
    """One text prompt on the first frame, propagated through the clip. Returns [T, H, W] uint16 masks, pixel
    value = object number (0 = background), the format of data/dataset/Annotations and data/predictions."""
    H, W = hw
    out = np.zeros((n_frames, H, W), np.uint16)
    sid = predictor.handle_request(dict(type='start_session', resource_path=str(resource)))['session_id']
    try:
        predictor.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text=text,
                                      output_prob_thresh=prob_thresh))
        for resp in predictor.handle_stream_request(dict(type='propagate_in_video', session_id=sid,
                                                         propagation_direction='forward',
                                                         output_prob_thresh=prob_thresh)):
            o = resp['outputs']
            for k, m in zip(np.asarray(o['out_obj_ids']).tolist(), np.asarray(o['out_binary_masks'])):
                m = np.asarray(m, bool)
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                out[resp['frame_index']][m] = min(int(k) + 1, 65535)
    finally:
        predictor.handle_request(dict(type='close_session', session_id=sid))
    return out


def frames_as_video(frames: list[Path], fps: float = 6) -> Path:
    """SAM 3 reads a video file; our clips are folders of JPEG frames. Write a temporary MP4 of the frames."""
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    tmp = Path(tempfile.mkdtemp(prefix='sam3_clip_')) / 'clip.mp4'
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return tmp


def cmd_clips(args):
    """Masks for every labelled clip of one split, written where the SAM 2.1 leaderboard reads them."""
    clips = json.loads((DATA / 'dataset' / 'clips.json').read_text())
    todo = [c for c, info in clips.items() if info.get('split') == args.split and info.get('status') == 'done']
    if not todo:
        sys.exit(f'No labelled {args.split} clips in {DATA / "dataset"}. Label clips with the SAM 2.1 pipeline first.')
    out_root = DATA / 'predictions' / args.name
    predictor = build_predictor(args.version, args.max_objects, args.checkpoint)
    for clip in todo:
        frames = sorted((DATA / 'dataset' / 'JPEGImages' / clip).glob('*.jpg'))
        h, w = cv2.imread(str(frames[0])).shape[:2]
        try:
            masks = track_text(predictor, frames[0].parent, args.text, (h, w), len(frames), args.prob)
        except Exception as e:                         # a folder of frames refused: give it a video file instead
            print(f'  frame folder refused ({type(e).__name__}); retrying with a temporary video')
            masks = track_text(predictor, frames_as_video(frames), args.text, (h, w), len(frames), args.prob)
        (out_root / clip).mkdir(parents=True, exist_ok=True)
        for f, m in zip(frames, masks):
            cv2.imwrite(str(out_root / clip / (f.stem + '.png')), m)
        print(f'{clip}: {len(frames)} frames, {len(np.unique(masks)) - 1} objects → {out_root / clip}')
    print(f'\nDone. Score it: cd ../sam2_pipeline && python fruit.py evaluate')
    print(f'(configs/lineup.yaml has the entry: type folder, path predictions/{args.name})')


def cmd_video(args):
    """SAM 3.1 on a stretch of one video: an overlay video and per-frame counts in data/results/sam3/."""
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f'cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    frames = []
    while len(frames) < int(args.seconds * args.fps):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        for _ in range(max(0, int(round(fps / args.fps)) - 1)):      # skip ahead to the next sampled frame
            cap.grab()
    cap.release()
    if not frames:
        sys.exit('no frames read')
    h, w = frames[0].shape[:2]
    tmp = Path(tempfile.mkdtemp(prefix='sam3_video_')) / 'stretch.mp4'
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()

    predictor = build_predictor(args.version, args.max_objects, args.checkpoint)
    masks = track_text(predictor, tmp, args.text, (h, w), len(frames), args.prob)

    out_dir = DATA / 'results' / 'sam3'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{Path(args.video).stem}_sam3'
    overlay = cv2.VideoWriter(str(out_dir / f'{stem}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    seen = set()
    with (out_dir / f'{stem}.csv').open('w', newline='', encoding='utf-8') as f:
        log = csv.writer(f)
        log.writerow(['frame', 'visible_now', 'ids_seen'])
        for i, (frame, m) in enumerate(zip(frames, masks), 1):
            ids = [k for k in np.unique(m) if k]
            seen.update(ids)
            tint = frame.copy()
            for k in ids:
                tint[m == k] = PALETTE[int(k) % len(PALETTE)]
            frame = cv2.addWeighted(tint, 0.35, frame, 0.65, 0)
            cv2.putText(frame, f'clusters visible: {len(ids)}  ids seen: {len(seen)}', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            overlay.write(frame)
            log.writerow([i, len(ids), len(seen)])
    overlay.release()
    print(f'overlay video: {out_dir / stem}.mp4\ncounts: {out_dir / stem}.csv')
    print('ids seen != verified unique clusters, and a cluster count is not a yield estimate.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--text', default='grape cluster', help='the text prompt (one concept phrase)')
    p.add_argument('--version', default='sam3.1', choices=['sam3.1', 'sam3'])
    p.add_argument('--checkpoint', type=Path, default=None,
                   help='local weights, e.g. model/sam3.1_multiplex.pt (default: the package downloads them)')
    p.add_argument('--max-objects', type=int, default=64, help='SAM 3.1 object capacity; 16 is too few for a vine row')
    p.add_argument('--prob', type=float, default=0.5, help='output probability threshold')
    p.add_argument('--data', type=Path, default=None,
                   help='the project folder with dataset/ and videos/ (default: $PROJECT_DIR, else the repository\'s data/)')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('clips', help='masks for the labelled clips → data/predictions/')
    s.add_argument('--split', default='val', choices=['val', 'train'])
    s.add_argument('--name', default='sam3_text', help='folder name under data/predictions/')
    s = sub.add_parser('video', help='overlay video and counts for a stretch of one video → data/results/sam3/')
    s.add_argument('video', type=Path)
    s.add_argument('--start', type=float, default=0, help='seconds into the video')
    s.add_argument('--seconds', type=float, default=10)
    s.add_argument('--fps', type=float, default=6, help='frames per second to run (lower is faster)')
    args = p.parse_args()
    global DATA
    if args.data:
        DATA = args.data.expanduser().resolve()
    if args.checkpoint is None:
        local = sorted(MODEL_DIR.glob('*.pt'), key=lambda f: f.stat().st_mtime)     # the newest
        if local:
            args.checkpoint = local[-1]
            print(f'using weights {args.checkpoint}')
    {'clips': cmd_clips, 'video': cmd_video}[args.cmd](args)


if __name__ == '__main__':
    main()
