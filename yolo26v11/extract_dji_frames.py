import argparse
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--every-sec', type=float, default=5)
    p.add_argument('--max-images', type=int, default=60)
    args = p.parse_args()
    if args.every_sec <= 0 or args.max_images < 1:
        p.error('Need positive --every-sec and --max-images')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        p.error('Video could not be opened: ' + str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        p.error('Video has invalid FPS')
    output = ROOT / 'data' / 'dji_to_label' / args.video.stem
    output.mkdir(parents=True, exist_ok=True)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(args.every_sec * fps))
    n = 0
    for index in range(0, total, step):
        if n >= args.max_images:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(output / f'{args.video.stem}_f{index:07d}.jpg'), frame)
        n += 1
    cap.release()
    print(f'Extracted {n} images to {output}')
    print('Label grape CLUSTERS manually; keep separate labeled test images from the same video session.')

if __name__ == '__main__':
    main()
