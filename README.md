# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

Our work for **CMSE 495 — Experiential Learning in Data Science** at Michigan State University, with our community partner **National Grape** and MSU AgBioResearch.

The project builds a low-cost, scalable computer-vision pipeline that estimates fruit yield from GoPro, phone and drone video of vineyards: find every grape cluster, outline it, follow it through the video, and count. We compare three model families on the same held-out clips with the same rules, starting with grapes and aiming to test other fruit (blueberries, apples, peaches) later.

## Start here

Three pipelines, three folders. Each one is self-contained: its own install guide, its own Python environment, and a `model/` folder with the one model it ships. They share one `data/` folder.

| Pipeline | What it does | Runs on | Start with |
|---|---|---|---|
| **SAM 2.1** — [`sam2_pipeline/`](sam2_pipeline/) | Label grape clusters in our videos with SAM 2.1's help (one box per cluster, SAM draws and follows the mask), fine-tune SAM 2.1 on those labels, and **score every model on the same held-out clips**. The leaderboard that compares all three pipelines lives here. | Laptop (CPU), Google Colab, MSU HPCC | [`sam2_pipeline/INSTALL.md`](sam2_pipeline/INSTALL.md): install, then a 10-minute end-to-end demo on public grapes. |
| **SAM 3.1** — [`sam3_pipeline/`](sam3_pipeline/) | Meta's newest Segment Anything, driven by a **text prompt** ("grape cluster"): no boxes, no training. Writes its masks where the leaderboard scores them, and runs on video. | NVIDIA GPU only (Colab, HPCC); gated weights | [`sam3_pipeline/README.md`](sam3_pipeline/README.md). Written, not yet run. |
| **YOLO26** — [`yolo26_pipeline/`](yolo26_pipeline/) | Train **YOLO26n-seg** on the public WGISD grape dataset, evaluate it, count clusters, and run it on video with optional tracking. Fast enough for whole videos. | Laptop (CPU), any GPU | [`yolo26_pipeline/INSTALL.md`](yolo26_pipeline/INSTALL.md): install, then the demo on public images and a short video. |

## What is in this repository

```
Capstone-MSU_Fruit/
├── README.md               this file
├── LICENSE.txt
├── data/                   everything the pipelines read and write (data/README.md)
│   ├── wgisd/              public WGISD grape dataset (in git)
│   ├── wgisd_yolo/         WGISD as YOLO labels (in git)
│   ├── results/            what the scripts write: leaderboards, counts, annotated videos (only yolo11/ in git)
│   ├── videos/             the partner's video (not in git: team drive)
│   └── dataset/, predictions/, models/, runs/, demo/   our labelled clips, each model's masks, training runs (not in git)
├── sam2_pipeline/          code, docs, environment and model/ of the SAM 2.1 pipeline
├── sam3_pipeline/          code, docs and model/ of the SAM 3.1 pipeline
├── yolo26_pipeline/        code, docs, environment and model/ of the YOLO26 pipeline
└── hpcc/                   run_all.sh: fine-tune every model on our clips and score them all, as HPCC batch jobs
```

Three rules keep it simple: **code lives in `<pipeline>/`**; **everything the code reads or produces lives in `data/`**; **`<pipeline>/model/` holds the one model that pipeline ships** (training runs go to `data/runs/`, and the winner is copied into `model/` with a note in `model/README.md` on how it was trained and how it scored).

**Where to find things:** inputs in `data/videos/` (partner video) and `data/dataset/` (our labelled clips); every result in `data/results/` (the leaderboard of all models at the top, `sam2/`, `sam3/`, `yolo26/` for each pipeline's own runs); trained weights in `data/models/` and `data/runs/`, the shipped ones in `<pipeline>/model/`. [`data/README.md`](data/README.md) has the full command-by-command table of what is written where, what is in git, and where the partner's video comes from.

## Main goals

- Preprocess image and video data
- Detect fruit and fruit clusters
- Segment detected fruit
- Count fruit from images and video
- Evaluate model performance
- Compare performance across fruit types and camera systems
- Create visualizations of the results

## Data

Image and video data collected from vineyards and orchards, including GoPro, phone and DJI drone footage, supplied by National Grape. Public data: the Embrapa WGISD grape dataset (CC BY-NC 4.0).

## Tools

Python · PyTorch · Segment Anything (SAM 2.1, SAM 3.1) · Ultralytics YOLO (YOLO26, YOLOE) · OpenCV · Jupyter / Google Colab · MSU HPCC

## Course

**CMSE 495 — Experiential Learning in Data Science**, Michigan State University. Team MSU FRUIT.
