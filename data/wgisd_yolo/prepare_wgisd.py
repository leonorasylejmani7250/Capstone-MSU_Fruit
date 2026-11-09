"""Build this folder from the public WGISD dataset (../wgisd): YOLO segmentation labels, split train / val / test, and
grape.yaml. The result is committed, so this only needs running to rebuild it, or to look at the labels:

    python prepare_wgisd.py                     rebuild images/, labels/ and grape.yaml
    python prepare_wgisd.py --preview 8         also draw the labels on 8 val photos → preview/
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent          # data/wgisd_yolo
SOURCE = HERE.parent / 'wgisd'                  # the WGISD dataset as published
YAML = HERE / 'grape.yaml'


def read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. The WGISD dataset should be in {SOURCE}')
    names = [Path(line.split()[0].replace('\\', '/')).stem
             for raw in path.read_text(encoding='utf-8').splitlines()
             if (line := raw.strip()) and not line.startswith('#')]
    if not names or len(set(names)) != len(names):
        raise ValueError(f'Empty or duplicate split entries in: {path}')
    return names


def polygon_label(mask: np.ndarray, minimum_area: float = 8.0) -> str | None:
    """One instance mask → one YOLO polygon line (the largest contour of a fragmented mask)."""
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < minimum_area:
        return None
    eps = max(0.5, 0.0015 * cv2.arcLength(contour, True))
    pts = cv2.approxPolyDP(contour, epsilon=eps, closed=True).reshape(-1, 2)
    if len(pts) < 3:
        return None
    h, w = mask.shape
    xy = np.clip(pts.astype(np.float64) / np.array([w, h]), 0, 1)
    return '0 ' + ' '.join(f'{v:.6f}' for v in xy.reshape(-1))


def convert(source: Path, dest: Path, val_fraction: float, seed: int) -> dict:
    """WGISD's train_masked / test_masked lists → images/<split>, labels/<split>. Part of train becomes val."""
    train_names = read_names(source / 'train_masked.txt')
    test_names = read_names(source / 'test_masked.txt')
    if set(train_names) & set(test_names):
        raise ValueError('WGISD train/test overlap: refusing to proceed')
    rng = random.Random(seed)
    rng.shuffle(train_names)
    n_val = max(1, round(len(train_names) * val_fraction))
    splits = {'train': train_names[n_val:], 'val': train_names[:n_val], 'test': test_names}

    summary = {}
    for split, names in splits.items():
        img_out, lbl_out = dest / 'images' / split, dest / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        total_polygons, skipped = 0, 0
        for stem in names:
            imgpath, maskpath = source / 'data' / f'{stem}.jpg', source / 'data' / f'{stem}.npz'
            if not imgpath.is_file() or not maskpath.is_file():
                raise FileNotFoundError(f'Missing WGISD pair for {stem}')
            img = cv2.imread(str(imgpath))
            if img is None:
                raise ValueError(f'Cannot read {imgpath}')
            with np.load(maskpath, allow_pickle=False) as loaded:
                key = 'arr_0' if 'arr_0' in loaded.files else loaded.files[0]
                masks = loaded[key]
            if masks.ndim != 3 or masks.shape[2] == 0:
                raise ValueError(f'Unexpected mask dimensions: {maskpath}: {masks.shape}')
            if abs(masks.shape[1] / masks.shape[0] - img.shape[1] / img.shape[0]) > 0.015:
                raise ValueError(f'Image/mask aspect ratios disagree: {stem}')
            rows = [polygon_label(masks[:, :, i]) for i in range(masks.shape[2])]
            skipped += sum(row is None for row in rows)
            rows = [row for row in rows if row is not None]
            if not rows:
                raise ValueError(f'No usable polygons in {stem}')
            shutil.copy2(imgpath, img_out / imgpath.name)
            (lbl_out / f'{stem}.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            total_polygons += len(rows)
        summary[split] = {'images': len(names), 'polygons': total_polygons, 'skipped': skipped}
        print(f'{split}: {len(names)} images, {total_polygons} polygons, {skipped} skipped')
    return summary


def write_yaml(dest: Path, summary: dict) -> Path:
    counts = ' / '.join(f"{s} {v['images']}" for s, v in summary.items())
    text = (f'# The public WGISD grape dataset as YOLO segmentation labels: images/ and labels/, split {counts}.\n'
            '# Written by prepare_wgisd.py. Paths are relative to this file.\n'
            'train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: grape_cluster\n')
    (dest / 'grape.yaml').write_text(text, encoding='utf-8')
    return dest / 'grape.yaml'


def preview(dest: Path, split: str, limit: int) -> Path:
    """Draw the polygons of `limit` photos of one split into preview/."""
    out = dest / 'preview'
    out.mkdir(exist_ok=True)
    n = 0
    for image_path in sorted((dest / 'images' / split).glob('*.jpg')):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for line in (dest / 'labels' / split / (image_path.stem + '.txt')).read_text().splitlines():
            arr = [float(x) for x in line.split()[1:]]
            if len(arr) < 6 or len(arr) % 2:
                raise ValueError('Malformed label: ' + line)
            xy = np.asarray(arr, dtype=np.float32).reshape(-1, 2) * [w, h]
            cv2.polylines(frame, [xy.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(str(out / image_path.name), frame)
        n += 1
        if n >= limit:
            break
    print(f'{n} photos with their labels drawn: {out}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--source', type=Path, default=SOURCE, help='the WGISD dataset (default: ../wgisd)')
    p.add_argument('--val-fraction', type=float, default=0.15, help='share of the WGISD train photos held out as val')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--preview', type=int, default=0, metavar='N', help='also draw the labels on N photos → preview/')
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='split for --preview')
    args = p.parse_args()
    if not 0 < args.val_fraction < 0.5:
        p.error('--val-fraction must be between 0 and 0.5')
    summary = convert(args.source.resolve(), HERE, args.val_fraction, args.seed)
    print('Written:', write_yaml(HERE, summary))
    if args.preview:
        preview(HERE, args.split, args.preview)


if __name__ == '__main__':
    main()
