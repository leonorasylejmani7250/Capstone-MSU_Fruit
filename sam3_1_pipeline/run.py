"""Run the model with a text prompt ("grape cluster") on our labelled clips or on a video, and write the result to ../data/.

    python run.py clips                                  masks for the held-out val clips → ../data/predictions/<name>/<clip>/<frame>.png
                                                         (then `python fruit.py evaluate` in ../sam2_pipeline scores them with every other model)
    python run.py video ../data/videos/row3.mp4          count the clusters in a video: overlay video, per-frame counts, the total
    python run.py video ../data/videos/CodyKresta_1025   every video in a folder

A video goes through the model in stretches of CHUNK_SECONDS that share one frame; a cluster keeps its id from one
stretch to the next when its masks on the shared frame overlap. The total is the number of ids seen in at least
MIN_FRAMES frames; a whole video adds it to ../data/results/cluster_counts.csv, the file every pipeline writes to.

Weights: the newest .pt in model/ (a fine-tuned model from finetune.py, or the release downloaded there), else Meta's release.
--checkpoint release forces Meta's weights; --checkpoint <file> picks a file. Options go before or after the command name.
Needs Meta's `sam3` package and a CUDA GPU (README.md).
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import pipeline as P

PALETTE = [(100, 210, 85), (215, 145, 68), (180, 85, 200), (76, 185, 230), (215, 110, 165), (96, 200, 175)]
CHUNK_SECONDS = 10      # a video goes through the model in stretches of this length
MIN_FRAMES = 2          # an id counts as a cluster once it has been seen in this many frames
MAX_SIDE = 1280         # frames are shrunk to this first (the model works at 1008 px)
VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.mts', '.webm'}


def pick_checkpoint(arg: str | None) -> Path | None:
    """None = Meta's release (downloaded by the package); else a local file."""
    if arg == 'release':
        return None
    if arg:
        f = Path(arg).expanduser()
        if not f.is_file():
            sys.exit(f'no such checkpoint: {f}')
        return f
    local = sorted(P.MODEL_DIR.glob('*.pt'), key=lambda f: f.stat().st_mtime)      # the newest file wins
    return local[-1] if local else None


def weights_label(checkpoint: Path | None) -> tuple[str, str]:
    """(folder name under predictions/, leaderboard row name) for a checkpoint."""
    recipe = checkpoint.with_suffix('.json') if checkpoint else None
    if recipe and recipe.exists():                                   # written by finetune.py
        run = json.loads(recipe.read_text()).get('run', checkpoint.stem)
        return f'{P.NAME}_{run}', f"{P.MODEL_NAME} · text '{{text}}' → fine-tuned on our clips ({run})"
    return f'{P.NAME}_text', f"{P.MODEL_NAME} · text '{{text}}' · zero-shot"


def build_predictor(max_num_objects: int, checkpoint: Path | None):
    """Meta's video predictor. SAM 3.1 adds Object Multiplex (many objects per clip); SAM 3 is the original.
    Two things are set for SAM 3.1, so that it runs with the packages of requirements.txt and with Meta's code as it is:
      - use_fa3=False: its attention layers use PyTorch's own attention. Meta's default calls FlashAttention 3 (the
        module flash_attn_interface), a separate package.
      - a session starts without the arguments the SAM 3.1 model does not take. Meta's predictor passes
        offload_state_to_cpu to every model, and only the SAM 3 model accepts it."""
    import sam3.model_builder as mb
    args = {'checkpoint_path': str(checkpoint)} if checkpoint else {}
    if P.VERSION != 'sam3.1':
        return mb.build_sam3_video_predictor(**args)
    predictor = mb.build_sam3_multiplex_video_predictor(max_num_objects=max_num_objects, use_fa3=False, **args)
    init_state = predictor.model.init_state
    known = inspect.signature(init_state).parameters
    predictor.model.init_state = lambda **kwargs: init_state(**{k: v for k, v in kwargs.items() if k in known})
    return predictor


def track_text(predictor, resource: Path, text: str, hw: tuple[int, int], n_frames: int, prob_thresh: float) -> np.ndarray:
    """One text prompt on the first frame, propagated through the clip. Returns [T, H, W] uint16 masks, pixel
    value = object number (0 = background), the format of data/dataset/Annotations and data/predictions."""
    H, W = hw
    out = np.zeros((n_frames, H, W), np.uint16)
    sid = predictor.handle_request(dict(type='start_session', resource_path=str(resource)))['session_id']
    try:
        predictor.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text=text, output_prob_thresh=prob_thresh))
        for resp in predictor.handle_stream_request(dict(type='propagate_in_video', session_id=sid,
                                                         propagation_direction='forward', output_prob_thresh=prob_thresh)):
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
    """The package reads a video file; our clips are folders of JPEG frames. Write a temporary MP4 of the frames."""
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    tmp = Path(tempfile.mkdtemp(prefix=f'{P.NAME}_clip_')) / 'clip.mp4'
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return tmp


def cmd_clips(args, data: Path, checkpoint: Path | None):
    """Masks for every labelled clip of one split, written where the SAM 2.1 leaderboard reads them."""
    clips = json.loads((data / 'dataset' / 'clips.json').read_text())
    todo = [c for c, info in clips.items() if info.get('split') == args.split and info.get('status') == 'done']
    if not todo:
        sys.exit(f'No labelled {args.split} clips in {data / "dataset"}. Label clips with the SAM 2.1 pipeline first.')
    folder, row = weights_label(checkpoint)
    out_root = data / 'predictions' / (args.name or folder)
    predictor = build_predictor(args.max_objects, checkpoint)
    for clip in todo:
        frames = sorted((data / 'dataset' / 'JPEGImages' / clip).glob('*.jpg'))
        h, w = cv2.imread(str(frames[0])).shape[:2]
        try:
            masks = track_text(predictor, frames[0].parent, args.text, (h, w), len(frames), args.prob)
        except Exception as e:                        # a folder of frames refused: give it a video file instead
            print(f'  frame folder refused ({type(e).__name__}); retrying with a temporary video')
            masks = track_text(predictor, frames_as_video(frames), args.text, (h, w), len(frames), args.prob)
        (out_root / clip).mkdir(parents=True, exist_ok=True)
        for f, m in zip(frames, masks):
            cv2.imwrite(str(out_root / clip / (f.stem + '.png')), m)
        print(f'{clip}: {len(frames)} frames, {len(np.unique(masks)) - 1} objects → {out_root / clip}')
    (out_root / 'recipe.json').write_text(json.dumps({                      # fruit.py evaluate reads this to name the row
        'row': row.format(text=args.text), 'pipeline': P.NAME, 'model': P.MODEL_NAME, 'text': args.text,
        'weights': str(checkpoint) if checkpoint else f"Meta's release ({P.HF_REPO})",
        'split': args.split, 'clips': todo, 'written': time.strftime('%Y-%m-%d %H:%M')}, indent=1))
    print(f'\nDone: {out_root}\nScore it: cd ../sam2_pipeline && python fruit.py evaluate   (row: {row.format(text=args.text)})')


def stitch(shared: np.ndarray | None, masks: np.ndarray, next_id: int) -> tuple[np.ndarray, int]:
    """Give the objects of a new stretch their ids for the whole video. `shared` is the frame both stretches hold, as
    the earlier one numbered it: an object whose mask on that frame overlaps an earlier object's (IoU 0.5 or more)
    takes its id, every other object the next free id. Returns (masks with the new ids, the next free id)."""
    lut = np.zeros(int(masks.max()) + 1, np.int64)
    if shared is not None:
        first = masks[0]
        both = (shared > 0) & (first > 0)
        if both.any():
            pairs, inter = np.unique(np.stack([shared[both], first[both]]), axis=1, return_counts=True)
            area_old, area_new = np.bincount(shared.ravel()), np.bincount(first.ravel(), minlength=len(lut))
            for i in np.argsort(-inter):
                old, new = int(pairs[0, i]), int(pairs[1, i])
                if inter[i] / (area_old[old] + area_new[new] - inter[i]) >= 0.5 and not lut[new]:
                    lut[new] = old
    for k in np.unique(masks):
        if k and not lut[k]:
            lut[k] = next_id
            next_id += 1
    return lut[masks], next_id


def video_key(video: Path, data: Path) -> str:
    """How a video is named in the count table: its path under videos/, else its file name."""
    videos = (data / 'videos').resolve()
    return video.resolve().relative_to(videos).as_posix() if video.resolve().is_relative_to(videos) else video.name


def record_count(video: Path, data: Path, model: str, total: int, seconds: float) -> Path:
    """One line per counted video in results/cluster_counts.csv, the file every pipeline adds its totals to."""
    table = data / 'results' / 'cluster_counts.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    new = not table.exists() or not table.stat().st_size
    with table.open('a', newline='', encoding='utf-8') as f:
        if new:
            csv.writer(f).writerow(['video', 'model', 'clusters', 'seconds', 'pipeline', 'written'])
        csv.writer(f).writerow([video_key(video, data), model, total, round(seconds, 1), P.NAME, time.strftime('%Y-%m-%d %H:%M')])
    return table


def count_video(video: Path, args, data: Path, predictor, model: str) -> None:
    """One video (or a stretch of it): overlay video, per-frame counts and the total in results/<pipeline>/."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {video}')
    stride = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 30) / args.fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    limit = int(args.seconds * args.fps) if args.seconds else 10**9
    per_chunk = max(2, int(CHUNK_SECONDS * args.fps))

    def read(n: int) -> list[np.ndarray]:
        frames = []
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                break
            scale = MAX_SIDE / max(frame.shape[:2])
            frames.append(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame)
            for _ in range(stride - 1):                               # skip ahead to the next sampled frame
                cap.grab()
        return frames

    out_dir = data / 'results' / P.NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{video.stem}_{P.NAME}'
    tmp = Path(tempfile.mkdtemp(prefix=f'{P.NAME}_video_')) / 'stretch.mp4'
    overlay, seen, shared, last, next_id, done = None, {}, None, None, 1, 0
    with (out_dir / f'{stem}.csv').open('w', newline='', encoding='utf-8') as f:
        log = csv.writer(f)
        log.writerow(['frame', 'visible_now', 'clusters_so_far'])
        while done < limit:
            frames = ([last] if last is not None else []) + read(min(per_chunk, limit - done))
            if len(frames) < (2 if last is not None else 1):
                break
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
            for frame in frames:
                vw.write(frame)
            vw.release()
            masks, next_id = stitch(shared, track_text(predictor, tmp, args.text, (h, w), len(frames), args.prob), next_id)
            if overlay is None:
                overlay = cv2.VideoWriter(str(out_dir / f'{stem}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
            for frame, m in list(zip(frames, masks))[0 if last is None else 1:]:        # the shared frame was written already
                ids = [int(k) for k in np.unique(m) if k]
                for k in ids:
                    seen[k] = seen.get(k, 0) + 1
                total = sum(1 for n in seen.values() if n >= MIN_FRAMES)
                tint = frame.copy()
                for k in ids:
                    tint[m == k] = PALETTE[k % len(PALETTE)]
                frame = cv2.addWeighted(tint, 0.35, frame, 0.65, 0)
                cv2.putText(frame, f'in view {len(ids)}  clusters so far {total}', (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                overlay.write(frame)
                done += 1
                log.writerow([done, len(ids), total])
            shared, last = masks[-1], frames[-1]
            print(f'  {done} frames, clusters so far {sum(1 for n in seen.values() if n >= MIN_FRAMES)}', flush=True)
    whole = not args.start and not args.seconds and not cap.grab()
    cap.release()
    tmp.unlink(missing_ok=True)
    if overlay is None:
        raise RuntimeError(f'no frames read from {video}')
    overlay.release()
    total = sum(1 for n in seen.values() if n >= MIN_FRAMES)
    print(f'overlay video: {out_dir / stem}.mp4\ncounts: {out_dir / stem}.csv')
    print(f'Clusters counted: {total} (ids seen in at least {MIN_FRAMES} frames)')
    if whole:
        print(f'Total written to {record_count(video, data, model, total, done / args.fps)}')


def cmd_video(args, data: Path, checkpoint: Path | None):
    """Count the clusters in one video, or in every video of a folder."""
    if not args.video.exists():
        sys.exit(f'cannot find {args.video}')
    model = weights_label(checkpoint)[1].format(text=args.text)
    videos = [args.video]
    if args.video.is_dir():
        videos = sorted(f for f in args.video.rglob('*') if f.is_file() and f.suffix.lower() in VIDEO_EXTS
                        and not any(part.startswith(('.', '~$')) for part in f.relative_to(args.video).parts))
        if not videos:
            sys.exit(f'No videos under {args.video}')
    table, done = data / 'results' / 'cluster_counts.csv', set()
    if args.video.is_dir() and table.exists() and not args.overwrite:      # a folder: the videos with a total are skipped
        with table.open(newline='', encoding='utf-8') as f:
            done = {row['video'] for row in csv.DictReader(f) if row['model'] == model}
    predictor = build_predictor(args.max_objects, checkpoint)
    for i, video in enumerate(videos, 1):
        if video_key(video, data) in done:
            print(f'[{i}/{len(videos)}] {video.name}: counted already, skipping (--overwrite to redo)')
            continue
        if len(videos) > 1:
            print(f'[{i}/{len(videos)}] {video}', flush=True)
        try:
            count_video(video, args, data, predictor, model)
        except RuntimeError as e:                                     # one unreadable file does not stop the folder
            if len(videos) == 1:
                sys.exit(str(e))
            print(f'  skipped: {e}')
    print('A count is the number of ids the model followed, not a hand count, and not a yield estimate.')


def general_options(parser: argparse.ArgumentParser, defaults: bool) -> None:
    """The options of both commands. The main parser gets them with their defaults, each command without, so that they
    work before or after the command name: `run.py --checkpoint release clips` and `run.py clips --checkpoint release`."""
    d = (lambda value: value) if defaults else (lambda value: argparse.SUPPRESS)
    parser.add_argument('--text', default=d('grape cluster'), help='the text prompt (one concept phrase)')
    parser.add_argument('--checkpoint', default=d(None),
                        help="weights: a .pt file, or 'release' for Meta's (default: the newest .pt in model/, else the release)")
    parser.add_argument('--max-objects', type=int, default=d(64), help='object capacity per clip; 16 is too few for a vine row')
    parser.add_argument('--prob', type=float, default=d(0.5), help='output probability threshold')
    parser.add_argument('--data', type=Path, default=d(None),
                        help="the project folder with dataset/ and videos/ (default: $PROJECT_DIR, else the repository's data/)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    general_options(p, defaults=True)
    after = argparse.ArgumentParser(add_help=False)
    general_options(after, defaults=False)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('clips', parents=[after], help='masks for the labelled clips → ../data/predictions/')
    s.add_argument('--split', default='val', choices=['val', 'train'])
    s.add_argument('--name', default=None, help='folder name under predictions/ (default: from the weights)')
    s = sub.add_parser('video', parents=[after], help='count the clusters in a video: overlay video, counts and the total → ../data/results/')
    s.add_argument('video', type=Path, help='a video, or a folder: every video in it')
    s.add_argument('--start', type=float, default=0, help='seconds into the video')
    s.add_argument('--seconds', type=float, default=0, help='how much of it (default 0: to the end; 20 for a quick look)')
    s.add_argument('--fps', type=float, default=6, help='frames per second to run (lower is faster)')
    s.add_argument('--overwrite', action='store_true', help='with a folder: count again the videos that have a total')
    args = p.parse_args()
    data = (args.data.expanduser().resolve() if args.data else P.DATA)
    checkpoint = pick_checkpoint(args.checkpoint)
    print(f"{P.MODEL_NAME} · weights: {checkpoint or 'Meta''s release'} · project folder: {data}")
    {'clips': cmd_clips, 'video': cmd_video}[args.cmd](args, data, checkpoint)


if __name__ == '__main__':
    main()
