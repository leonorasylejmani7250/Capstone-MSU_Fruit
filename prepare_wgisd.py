from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Check that WGISD has been moved to external/wgisd")
    names = [Path(line.split()[0].replace('\\', '/')).stem
             for raw in path.read_text(encoding='utf-8').splitlines()
             if (line := raw.strip()) and not line.startswith('#')]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Empty or duplicate split entries in: {path}")
    return names


def polygon_label(mask: np.ndarray, minimum_area: float = 8.0) -> str | None:
    """One instance -> one YOLO polygon (largest contour for fragmented masks)."""
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
    train_names = read_names(source / 'train_masked.txt')
    test_names = read_names(source / 'test_masked.txt')
    if set(train_names) & set(test_names):
        raise ValueError('WGISD train/test overlap: refusing to proceed')
    rng = random.Random(seed)
    rng.shuffle(train_names)
    n_val = max(1, round(len(train_names) * val_fraction))
    splits = {'train': train_names[n_val:], 'val': train_names[:n_val], 'test': test_names}

    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, names in splits.items():
        img_out = dest / 'images' / split
        lbl_out = dest / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        total_polygons, skipped = 0, 0
        for stem in names:
            imgpath = source / 'data' / f'{stem}.jpg'
            maskpath = source / 'data' / f'{stem}.npz'
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT / 'external' / 'wgisd')
    p.add_argument('--val-fraction', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 0 < args.val_fraction < 0.5:
        p.error('--val-fraction must be between 0 and 0.5')
    output = ROOT / 'data' / 'wgisd_seg'
    convert(args.source.resolve(), output, args.val_fraction, args.seed)
    # Paths relative to grape.yaml itself (no 'path:' key), so the file works on every computer.
    cfg = {'train': 'data/wgisd_seg/images/train', 'val': 'data/wgisd_seg/images/val',
           'test': 'data/wgisd_seg/images/test', 'names': {0: 'grape_cluster'}}
    (ROOT / 'grape.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print('Created:', ROOT / 'grape.yaml')

if __name__ == '__main__':
    main()
