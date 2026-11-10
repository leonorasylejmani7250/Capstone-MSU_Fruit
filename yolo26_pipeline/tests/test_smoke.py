"""No-GPU checks: the folders this pipeline points at exist, the video renderer works, the dataset build works,
a weights file is described and the input size is chosen as documented.
Run from the pipeline folder:  python -m unittest discover -s tests -v
"""
from __future__ import annotations
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]                 # the pipeline folder
sys.path.insert(0, str(HERE))
import pipeline                                             # noqa: E402
sys.path.insert(0, str(pipeline.DATA / 'wgisd_yolo'))      # prepare_wgisd.py lives with the dataset

if 'ultralytics' not in sys.modules:                        # the renderer does not need the model itself
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=object)

from prepare_wgisd import convert   # noqa: E402
from video import clusters, pick, render      # noqa: E402


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
    def int(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.data
    def tolist(self): return self.data.tolist()


class FakeBoxes:
    def __init__(self, xyxy, conf, ids=None):
        self.xyxy, self.conf = FakeTensor(xyxy), FakeTensor(conf)
        self.id = FakeTensor(ids) if ids is not None else None
    def __len__(self):
        return len(self.xyxy.data)


class FakeResult:
    def __init__(self, boxes, polygons):
        self.boxes = boxes
        self.masks = types.SimpleNamespace(xy=polygons) if polygons else None


class SmokeTest(unittest.TestCase):
    def test_pipeline_paths(self):
        self.assertEqual(pipeline.ROOT, HERE)
        self.assertTrue(pipeline.WGISD.is_file(), f'missing {pipeline.WGISD}')
        for split in ('train', 'val', 'test'):
            self.assertTrue((pipeline.WGISD.parent / 'images' / split).is_dir())
        self.assertTrue(pipeline.MODEL.is_file(), f'missing {pipeline.MODEL}')
        self.assertEqual(pipeline.RUNS.parent.name, 'runs')
        self.assertEqual(pipeline.RESULTS.parent.name, 'results')
        self.assertEqual(pipeline.RUNS.name, pipeline.NAME)

    def test_weights_are_described(self):
        trained = {'train_args': {'model': 'yolo26n-seg.pt', 'imgsz': 512, 'epochs': 50},
                   'train_metrics': {'metrics/mAP50(M)': 0.83}, 'train_results': {'epoch': list(range(50))}}
        line, warnings, size = pipeline.describe(trained, ['grape_cluster'], 'model/best.pt')
        self.assertEqual((warnings, size), ([], 512))
        self.assertIn('50 epoch(s) at 512 px', line)
        test_run = {'train_args': {'imgsz': 512, 'epochs': 1}, 'train_metrics': {'metrics/mAP50(M)': 0.01},
                    'train_results': {'epoch': [1]}}
        self.assertIn('test run', pipeline.describe(test_run, ['grape_cluster'], 'runs/x/weights/best.pt')[1][0])
        with self.assertRaises(ValueError):                 # a base model has no grape class
            pipeline.describe({'train_args': {'imgsz': 640}}, ['person', 'bicycle', 'car'], 'yolo26n-seg.pt')

    def test_input_size_is_where_the_model_is_most_sure(self):
        rows = [{'imgsz': 256, 'sure': 3.0, 'top_score': 0.9}, {'imgsz': 512, 'sure': 9.5, 'top_score': 0.95},
                {'imgsz': 1280, 'sure': 0.0, 'top_score': 0.2}]
        self.assertEqual(pick(rows, 768), 512)
        unsure = [{'imgsz': 256, 'sure': 0.0, 'top_score': 0.1}, {'imgsz': 512, 'sure': 0.0, 'top_score': 0.2}]
        self.assertEqual(pick(unsure, 768), 512)            # sure of nothing: the size with the top score
        self.assertEqual(pick([{'imgsz': 256, 'sure': 0.0, 'top_score': 0.0}], 768), 768)       # no box at all

    def test_wgisd_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, dest = root / 'source', root / 'converted'
            (source / 'data').mkdir(parents=True)
            names = [f'CFR_{i}' for i in range(8)]
            (source / 'train_masked.txt').write_text('\n'.join(names[:6]))
            (source / 'test_masked.txt').write_text('\n'.join(names[6:]))
            for name in names:
                img = np.zeros((80, 120, 3), np.uint8)
                binary = np.zeros((80, 120, 1), np.uint8)
                cv2.ellipse(binary[:, :, 0], (55, 40), (18, 28), 0, 0, 360, 255, -1)
                cv2.imwrite(str(source / 'data' / f'{name}.jpg'), img)
                np.savez_compressed(source / 'data' / f'{name}.npz', binary)
            summary = convert(source, dest, .20, 42)
            self.assertEqual({k: v['images'] for k, v in summary.items()}, {'train': 5, 'val': 1, 'test': 2})
            for split, stats in summary.items():
                self.assertEqual(stats['images'], stats['polygons'])
                for path in (dest / 'labels' / split).glob('*.txt'):
                    parts = path.read_text().strip().split()
                    self.assertEqual(parts[0], '0')
                    self.assertEqual((len(parts) - 1) % 2, 0)
                    self.assertGreaterEqual(len(parts), 7)
                    self.assertTrue(all(0 <= float(p) <= 1 for p in parts[1:]))

    def test_render_all_detections_even_without_ids(self):
        frame = np.zeros((280, 540, 3), np.uint8)
        polys = [np.array([[15, 40], [115, 40], [115, 190], [15, 190]]),
                 np.array([[220, 50], [320, 50], [320, 210], [220, 210]])]
        boxes = FakeBoxes([[15, 40, 115, 190], [220, 50, 320, 210]], [.9, .3])
        seen = {}
        for _ in range(300):
            out, n, now = render(frame, FakeResult(boxes, polys), seen, .25, 'track')
            self.assertEqual((n, now, len(seen)), (2, 0, 0))
            self.assertTrue(np.any(out != frame))
        with_ids = FakeResult(FakeBoxes([[15, 40, 115, 190], [220, 50, 320, 210]], [.9, .3], [17]), polys)
        for frames in range(1, 31):                         # an id counts once it has been followed for 30 frames
            self.assertEqual(clusters(seen, 30), 0)
            out, n, now = render(frame, with_ids, seen, .25, 'track', 30)
            self.assertEqual((n, now, seen), (2, 1, {17: frames}))
        self.assertEqual(clusters(seen, 30), 1)


if __name__ == '__main__':
    unittest.main()
