# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

CMSE 495 (Michigan State University) capstone with National Grape and MSU AgBioResearch: a computer-vision pipeline that finds, outlines, follows and counts grape clusters in GoPro, phone and drone video of vineyards.

## Start here

Three pipelines, three folders, one shared `data/` folder. Each pipeline has its own install guide, its own Python environment and a `model/` folder with the model it ships.

| Pipeline | What it does | Runs on | Start with |
|---|---|---|---|
| **SAM 2.1** — [`sam2_pipeline/`](sam2_pipeline/) | Label clusters in our videos (one box each, SAM 2.1 draws and follows the mask), fine-tune SAM 2.1 on the labels, and score every model on the same held-out clips. The leaderboard lives here. | Laptop, Google Colab, MSU HPCC | [`sam2_pipeline/INSTALL.md`](sam2_pipeline/INSTALL.md) |
| **SAM 3.1** — [`sam3_pipeline/`](sam3_pipeline/) | Segment and track clusters from the text prompt "grape cluster", no training; its masks go to the leaderboard. | NVIDIA GPU (Colab, HPCC) | [`sam3_pipeline/README.md`](sam3_pipeline/README.md) |
| **YOLO26** — [`yolo26_pipeline/`](yolo26_pipeline/) | Train YOLO26n-seg on the public WGISD grape dataset, evaluate it, count clusters, run it on whole videos with optional tracking. | Laptop, any GPU | [`yolo26_pipeline/INSTALL.md`](yolo26_pipeline/INSTALL.md) |

Training and fine-tuning every model on the HPCC in one go: [`hpcc/README.md`](hpcc/README.md).

## Layout

```
Capstone-MSU_Fruit/
├── data/              everything the pipelines read and write (data/README.md)
│   ├── wgisd/         public WGISD grape dataset (in git)
│   ├── wgisd_yolo/    WGISD as YOLO labels (in git)
│   ├── videos/        the partner's video (team drive, not in git)
│   ├── dataset/       our labelled clips (not in git)
│   └── results/       every output: the leaderboard, then sam2/ sam3/ yolo26/
├── sam2_pipeline/     code, install guide, environment, model/
├── sam3_pipeline/     code, guide, model/
├── yolo26_pipeline/   code, install guide, environment, model/
└── hpcc/              run_all.sh: fine-tune and score every model as HPCC batch jobs
```

Three rules: code lives in `<pipeline>/`; everything it reads or writes lives in `data/`; `<pipeline>/model/` holds the model that pipeline ships. [`data/README.md`](data/README.md) lists what every command writes and where.

## Data

Vineyard video (GoPro, phone, DJI drone) from National Grape, on the team drive. Public data: the Embrapa WGISD grape dataset (CC BY-NC 4.0), in `data/wgisd/`.

## Tools

Python · PyTorch · Segment Anything (SAM 2.1, SAM 3.1) · Ultralytics YOLO (YOLO26, YOLOE) · OpenCV · Jupyter / Google Colab · MSU HPCC
