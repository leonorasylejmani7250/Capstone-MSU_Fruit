"""Compare the model's cluster counts with hand counts on the 27 WGISD test images (Michal's count test).

Prints the per-image error and the mean absolute error; writes data/results/yolo26/count_test.csv and the
annotated images to data/results/yolo26/count_test/. Default model: model/best.pt.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent       # yolo26_pipeline/
DATA = ROOT.parent / 'data'                  # the shared data folder of the repository
MODEL = ROOT / 'model' / 'best.pt'           # the model this pipeline ships
IMAGE_DIR = DATA / 'wgisd_yolo' / 'images' / 'test'

# (image, grape clusters counted by hand)
HAND_COUNTS = [
    ("SYH_2017-04-27_1322.jpg", 20),
    ("SYH_2017-04-27_1318.jpg", 8),
    ("SYH_2017-04-27_1304.jpg", 7),
    ("SYH_2017-04-27_1269.jpg", 11),
    ("SYH_2017-04-27_1239.jpg", 21),
    ("SVB_20180427_151818928.jpg", 21),
    ("SVB_1972.jpg", 22),
    ("SVB_1954.jpg", 19),
    ("SVB_1944.jpg", 18),
    ("SVB_1935.jpg", 25),
    ("CSV_20180427_144723166_HDR.jpg", 14),
    ("CSV_20180427_144535647.jpg", 13),
    ("CSV_20180427_144507419.jpg", 16),
    ("CSV_1898.jpg", 11),
    ("CSV_1877.jpg", 8),
    ("CFR_1666.jpg", 12),
    ("CFR_1651.jpg", 14),
    ("CFR_1643.jpg", 15),
    ("CFR_1641.jpg", 30),
    ("CFR_1639.jpg", 17),
    ("CFR_1638.jpg", 10),
    ("CDY_2054.jpg", 16),
    ("CDY_2051.jpg", 11),
    ("CDY_2046.jpg", 10),
    ("CDY_2043.jpg", 10),
    ("CDY_2040.jpg", 8),
    ("CDY_2015.jpg", 20),
]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', type=Path, default=MODEL, help='weights (default: model/best.pt)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=512)
    args = p.parse_args()
    if not args.model.is_file():
        p.error(f'Missing weights: {args.model} (train.py makes them; model/README.md says what goes in model/)')
    out_dir = DATA / 'results' / 'yolo26' / 'count_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Using model: {args.model}')
    model = YOLO(str(args.model))

    rows = []
    for filename, actual in HAND_COUNTS:
        image_path = IMAGE_DIR / filename
        if not image_path.is_file():
            print(f'Missing image: {image_path}')
            continue
        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        predicted = len(result.masks) if result.masks is not None else 0     # one mask per detected cluster
        error = predicted - actual
        rows.append({'image': filename, 'actual_count': actual, 'predicted_count': predicted,
                     'error': error, 'absolute_error': abs(error)})
        print(f'{filename:38} Actual: {actual:2} | Predicted: {predicted:2} | Error: {error:+3}')
        result.save(filename=str(out_dir / filename))          # the image with the predictions drawn on it

    if not rows:
        print('No images were evaluated.')
        return
    mae = sum(r['absolute_error'] for r in rows) / len(rows)
    exact = sum(r['actual_count'] == r['predicted_count'] for r in rows)
    print('\n--- Count test summary ---')
    print(f'Images evaluated: {len(rows)}/{len(HAND_COUNTS)}')
    print(f'Mean absolute count error: {mae:.2f} clusters/image')
    print(f'Exact count matches: {exact}/{len(rows)} ({100 * exact / len(rows):.1f}%)')
    print(f'Total counted by hand: {sum(r["actual_count"] for r in rows)}')
    print(f'Total predicted: {sum(r["predicted_count"] for r in rows)}')

    csv_path = out_dir.parent / 'count_test.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved to: {csv_path}')
    print(f'Annotated images saved to: {out_dir}')


if __name__ == '__main__':
    main()
