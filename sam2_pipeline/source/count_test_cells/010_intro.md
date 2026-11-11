# SAM 2.1 count test on the WGISD test photos

Runs the automatic SAM 2.1 pipeline (**detector → SAM 2.1**) on the 27 WGISD test photos, counts the grape clusters it finds, and compares the counts with the hand counts below. The detector's own count is shown next to it, so the table says what SAM 2.1 adds. It is the SAM 2.1 counterpart of `yolo26_pipeline/count_test.py`, on the same photos and the same hand counts.

**How to run:** the same way as `demo.ipynb` (see `INSTALL.md`): `conda activate ./envs`, then `jupyter notebook` **from this folder** (`sam2_pipeline/`), open this notebook and *Run ▸ Run All Cells*. The first run installs `ultralytics` into the environment and downloads the SAM 2.1 weights (about 150 MB). A laptop CPU needs a few minutes for the 27 photos.

Everything it writes goes to `data/count_test/` (git ignores it): the table in `results/count_test.csv`, the metrics in `results/count_metrics.csv`, one overlay image per photo in `results/overlays/`.
