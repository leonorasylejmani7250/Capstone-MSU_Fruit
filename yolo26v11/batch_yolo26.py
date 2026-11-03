from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from yolo26_common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description='YOLO26 batch inference on a video folder')
    parser.add_argument('--video-dir', type=Path, default=ROOT / 'videos' / 'CodyKresta_1025')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'yolo26_full')
    parser.add_argument('--mode', choices=['detect', 'track'], default='detect')
    parser.add_argument('--conf', type=float, default=0.30)
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--nms-free', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not args.video_dir.is_dir() or not args.model.is_file():
        parser.error('The input folder and trained model must exist.')
    candidates = sorted(
        f for f in args.video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}
    )
    if not candidates:
        parser.error('No video files found.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Found {len(candidates)} video(s)', flush=True)
    for i, video in enumerate(candidates, 1):
        output = args.output_dir / (video.stem + '_yolo26.mp4')
        csv_path = output.with_suffix('.csv')
        if output.is_file() and csv_path.is_file() and not args.overwrite:
            print(f'[{i}/{len(candidates)}] SKIP existing: {video.name}', flush=True)
            continue
        print(f'[{i}/{len(candidates)}] Processing {video.name}', flush=True)
        cmd = [
            sys.executable, str(ROOT / 'video_yolo26.py'),
            '--video', str(video), '--model', str(args.model),
            '--output', str(output), '--mode', args.mode,
            '--conf', str(args.conf), '--imgsz', str(args.imgsz), '--max-frames', '0',
        ]
        if args.nms_free:
            cmd.append('--nms-free')
        subprocess.run(cmd, check=True, cwd=ROOT)
    print('All requested videos completed.', flush=True)


if __name__ == '__main__':
    main()
