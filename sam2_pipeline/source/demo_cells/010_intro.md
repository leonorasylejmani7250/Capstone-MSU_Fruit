# MSU FRUIT · Demo: fine-tune SAM 2.1 on grape clusters and score it

This notebook previews the whole pipeline on public sample data, end to end: **clips → labels → score pretrained SAM 2.1 → fine-tune it → score again → look at the result**. It is the short version of `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb`, which does the same on our own vineyard video (with uploading and labelling).

- **Runs on a laptop CPU in about 5–10 minutes** (seconds per step on a GPU). It uses the smallest SAM 2.1 model (*tiny*) so it finishes without a GPU.
- **First run downloads about 165 MB**: Meta's SAM 2 code and the SAM 2.1 tiny weights, cached in `~/.fruit_sam2/`. The four grape photos with hand-drawn cluster outlines come from the copy of the Embrapa WGISD dataset (CC BY-NC 4.0) in `data/wgisd/`.
- **Everything it makes goes to `data/demo/`** in this repository (git ignores it). Delete that folder to start over.

Setup instructions: `INSTALL.md`. Run the cells top to bottom (**Run ▸ Run All Cells**).
