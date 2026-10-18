"""Draw the converted WGISD labels on a few images of one split and save them to outputs/label_preview for checking."""
import argparse
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--limit', type=int, default=8)
    args = p.parse_args()
    imgs = ROOT / 'data' / 'wgisd_seg' / 'images' / args.split
    labels = ROOT / 'data' / 'wgisd_seg' / 'labels' / args.split
    out = ROOT / 'outputs' / 'label_preview'
    out.mkdir(parents=True, exist_ok=True)
    found = 0
    for image_path in sorted(imgs.glob('*.jpg')):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        h,w = frame.shape[:2]
        text = (labels / (image_path.stem + '.txt')).read_text().splitlines()
        for line in text:
            arr = [float(x) for x in line.split()[1:]]
            if len(arr) < 6 or len(arr) % 2 != 0:
                raise ValueError('Malformed label: ' + line)
            xy = np.asarray(arr, dtype=np.float32).reshape(-1, 2) * [w, h]
            cv2.polylines(frame, [xy.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(str(out / image_path.name), frame)
        found += 1
        if found >= args.limit:
            break
    print(f'Saved {found} label preview images in: {out}')

if __name__ == '__main__':
    main()
