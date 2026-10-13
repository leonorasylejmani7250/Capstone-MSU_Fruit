## Appendix F · What was checked before this pipeline was handed over, and what was not

Checked on 12–13 Oct 2026 on a CPU-only Linux machine (Python 3.13, PyTorch 2.14, SAM 2 code at commit `2b90b9f`). The runs used the SAM 2.1 **tiny** model at 512 px so they finish on a CPU:

- **This notebook, top to bottom, in demo mode**: setup from scratch, demo download, clips, split, scoring four models, fine-tuning with Meta's trainer, re-scoring, the comparison video, and the run on video.
- **The HPCC command line (`fruit.py`) on a fresh project**:
  - clips from videos, with a whole-video split;
  - approving labels;
  - imports;
  - fine-tuning on video clips and single images together;
  - `evaluate` with every model type in one lineup file: SAM 2.1, YOLOE, YOLOE → SAM 2.1, a folder of PNGs, and a Python class. Fine-tuned rows were added automatically, and a "model" that copies the labels scored exactly 100;
  - `video`, `export-yolo`, `train-detector` and `status`.
- **Imports**:
  - YOLO box labels became SAM 2.1 masks matching the true WGISD masks at mean IoU 0.80 (36 clusters, tiny at 512 px);
  - a COCO box-only object matched at IoU 0.91;
  - outlines were kept as they are;
  - Roboflow-style file names were grouped by source video, and frames of one video never landed on both sides of the split.
- **Scoring**:
  - J (mask IoU) and F (boundary accuracy) agree exactly with Meta's own SA-V/DAVIS evaluator on 965 object-frames;
  - instance matching, in-view averaging and counting were checked against hand-worked cases;
  - scoring the labels against themselves gives 100.
- **Labelling**: boxes, corrective clicks, re-boxing, delete, undo, saving and approval. These were exercised both from code and by driving the real screen with a mouse in JupyterLab.

**Not checked, so treat as unproven until you have run it once:**

- **Anything on a GPU.** On a T4 or V100, training uses float16, because those GPUs lack bfloat16; it switches to bfloat16 by itself if the loss blows up. Speed and memory are estimates, which is why training retries with fewer frames when memory runs out. `base_plus` and `large` were not run at all. They differ from `tiny` only in backbone settings copied from Meta's own configs.
- **Colab itself**: the Drive mount, the upload button, the form fields, and the labelling screen inside Colab's widget system (tested in JupyterLab).
- **The HPCC itself**: `hpcc/setup_env.sh`, `hpcc/run.sb` and OnDemand Jupyter follow ICER's documentation as of October 2026, but were not run on the cluster.
- **`SAM3Text` and the SAM 3.1 snippet** in Appendix B (gated weights).
- **Whether fine-tuning helps on Riesling footage.** That is what Step 6 is for.
- The default training settings (learning rate, steps) are Meta's recipe scaled down, not tuned on vineyard video.
