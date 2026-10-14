## What this shows, and what comes next
- **The pipeline works end to end.** Clips, labels, scoring, Meta's trainer, re-scoring and comparison all ran on this computer.
- **The scores are not a result yet.** Twenty training steps on three sample clips cannot be expected to beat pretrained SAM 2.1; a gap of a point or two either way is noise. The real comparison runs on our Riesling video, with thousands of steps and enough held-out clips.
- **Next, with real data:** open `sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` in Google Colab (GPU), or use the HPCC scripts in `sam2_pipeline/hpcc/`. Upload videos, label them (one box per cluster; SAM 2.1 does the outlines), fine-tune, and score every model. `sam2_pipeline/README.md` walks through it.
