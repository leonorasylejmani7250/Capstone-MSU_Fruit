"""Lightweight, no-GPU smoke tests for conversion and rendering.
Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

# Stub model import when Ultralytics is not installed in the test environment.
if 'ultralytics' not in sys.modules:
    try:
        import ultralytics
    except ImportError:
        sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=object)

from prepare_wgisd import convert
from video import render


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
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.id = FakeTensor(ids) if ids is not None else None
    def __len__(self):
        return len(self.xyxy.data)


class FakeResult:
    def __init__(self, boxes, polygons):
        self.boxes = boxes
        self.masks = types.SimpleNamespace(xy=polygons) if polygons else None


class SmokeTest(unittest.TestCase):
    def test_prepare_split_and_polygons(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, dest = root / 'source', root / 'converted'
            (source/'data').mkdir(parents=True)
            names = [f'CFR_{i}' for i in range(8)]
            (source/'train_masked.txt').write_text('\n'.join(names[:6]))
            (source/'test_masked.txt').write_text('\n'.join(names[6:]))
            for name in names:
                img = np.zeros((80,120,3), np.uint8)
                binary = np.zeros((80,120,1), np.uint8)
                cv2.ellipse(binary[:,:,0], (55,40), (18,28), 0, 0, 360, 255, -1)
                cv2.imwrite(str(source/'data'/f'{name}.jpg'),img)
                np.savez_compressed(source/'data'/f'{name}.npz',binary)
            summary = convert(source, dest, .20, 42)
            self.assertEqual({k:v['images'] for k,v in summary.items()}, {'train':5,'val':1,'test':2})
            for split,stats in summary.items():
                self.assertEqual(stats['images'],stats['polygons'])
                for path in (dest/'labels'/split).glob('*.txt'):
                    parts = path.read_text().strip().split()
                    self.assertEqual(parts[0],'0')
                    self.assertEqual((len(parts)-1)%2,0)
                    self.assertGreaterEqual(len(parts),7)
                    self.assertTrue(all(0<=float(p)<=1 for p in parts[1:]))

    def test_render_all_detections_even_without_ids(self):
        frame = np.zeros((280,540,3), np.uint8)
        polys = [np.array([[15,40],[115,40],[115,190],[15,190]]),
                 np.array([[220,50],[320,50],[320,210],[220,210]])]
        boxes = FakeBoxes([[15,40,115,190],[220,50,320,210]],[.9,.3])
        seen = set()
        for i in range(300):
            result = FakeResult(boxes,polys)
            out,n,now = render(frame,result,seen,.25,'track')
            self.assertEqual((n,now,len(seen)),(2,0,0))
            self.assertTrue(np.any(out != frame))
        with_ids = FakeResult(FakeBoxes([[15,40,115,190],[220,50,320,210]],[.9,.3],[17]),polys)
        out,n,now = render(frame,with_ids,seen,.25,'track')
        self.assertEqual((n,now,len(seen)),(2,1,1))
        self.assertIn(17,seen)
        print('Render success: 300 frames with 2 no-ID detections; missing one ID handled.')

if __name__ == '__main__':
    unittest.main()
