"""Check that the install works, in about a minute.

Usage, from the repository folder with the environment active:
    python tests/test_install.py

It imports every package the pipeline needs, fetches Meta's SAM 2 code and the SAM 2.1 tiny weights on
the first run (about 160 MB, cached in ~/.fruit_sam2), then runs SAM 2.1 on a small synthetic clip: a
bright blob drifting across three frames, given as one box. It passes if SAM 2.1 outlines and follows
the blob, and if the scoring code gives a perfect score to a perfect prediction. Prints INSTALL OK.

It is a plain script, not a unittest module, so `python -m unittest discover -s tests` (the YOLO11
pipeline's test command) imports it without running it.
"""
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(REPO / "sam2_pipeline"))
    t0 = time.time()
    print("1/4 importing packages and fetching the SAM 2 code (first run only) ...", flush=True)
    import cv2
    import numpy as np
    from PIL import Image

    import fruit_sam2 as F

    print(f"    torch {F.torch.__version__} · {'GPU ' + F.GPU_NAME if F.DEVICE == 'cuda' else 'CPU'} · "
          f"SAM 2 code in {F.SAM2_REPO}")

    with tempfile.TemporaryDirectory() as tmp:
        F.use_project(Path(tmp) / "project")
        F.SAM2_SIZE = "tiny"
        F.SAM2_IMAGE_SIZE = 512

        print("2/4 making a 3-frame test clip ...", flush=True)
        clip_dir = F.IMG / "install_check"
        clip_dir.mkdir(parents=True)
        truth = np.zeros((3, 360, 480), np.uint16)
        for t in range(3):
            img = np.full((360, 480, 3), (40, 70, 40), np.uint8)                        # dark green background
            cv2.ellipse(img, (150 + 60 * t, 180), (70, 50), 0, 0, 360, (210, 230, 120), -1)   # pale green blob
            cv2.ellipse(truth[t], (150 + 60 * t, 180), (70, 50), 0, 0, 360, 1, -1)
            Image.fromarray(img).save(clip_dir / f"{t:05d}.jpg", quality=95)
        clip = F.Clip("install_check")

        print("3/4 loading SAM 2.1 tiny (downloads ~160 MB on the first run) and tracking the blob ...", flush=True)
        predictor = F.load_sam2("tiny")
        pred = F.sam2_track(predictor, clip, [dict(obj=1, frame=0, box=[80, 130, 220, 230])], backward=False)
        ious = [F.mask_iou(pred[t] == 1, truth[t] == 1) for t in range(3)]
        print("    mask overlap with the truth, frame by frame:", " ".join(f"{v:.2f}" for v in ious))
        assert min(ious) > 0.8, "SAM 2.1 did not follow the test blob: the install is broken"

        print("4/4 checking the scoring code ...", flush=True)
        rows = F.score_prompted(truth, truth, [dict(obj=1, frame=0)])
        assert abs(rows[0]["J"] - 100) < 1e-9 and abs(rows[0]["F"] - 100) < 1e-9

    print(f"INSTALL OK ({time.time() - t0:.0f} s). Next: jupyter notebook, then open demo.ipynb.")


if __name__ == "__main__":
    main()
