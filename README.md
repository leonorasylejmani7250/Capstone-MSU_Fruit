# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

CMSE 495 (Michigan State University) capstone with National Grape and MSU AgBioResearch: a computer-vision pipeline that finds, outlines, follows and counts grape clusters in GoPro, phone and drone video of vineyards.

## Start here

One folder per model, one shared `data/` folder. Each pipeline has its own install guide, its own Python environment and a `model/` folder with the model it ships.

| Pipeline | What it does | Runs on | Start with |
|---|---|---|---|
| **SAM 2.1** — [`sam2_pipeline/`](sam2_pipeline/) | Label clusters in our videos (one box each, SAM 2.1 draws and follows the mask), fine-tune SAM 2.1 on the labels, and score every model on the same held-out clips. The leaderboard lives here. | Laptop, Google Colab, MSU HPCC | [`sam2_pipeline/INSTALL.md`](sam2_pipeline/INSTALL.md) |
| **SAM 3** — [`sam3_pipeline/`](sam3_pipeline/) | Segment and track clusters from the text prompt "grape cluster" (zero-shot), and fine-tune it on our labelled clips; its masks go to the leaderboard. | NVIDIA GPU (Colab, HPCC) | [`sam3_pipeline/README.md`](sam3_pipeline/README.md) |
| **SAM 3.1** — [`sam3_1_pipeline/`](sam3_1_pipeline/) | The same with SAM 3.1, Meta's newer release with a faster multi-object tracker. | NVIDIA GPU (Colab, HPCC) | [`sam3_1_pipeline/README.md`](sam3_1_pipeline/README.md) |
| **YOLO26** — [`yolo26_pipeline/`](yolo26_pipeline/) | Train YOLO26n-seg on the public WGISD grape photos, score it, count clusters, run it on whole videos with optional tracking, fine-tune it on our labels. | Laptop, any GPU | [`yolo26_pipeline/INSTALL.md`](yolo26_pipeline/INSTALL.md) |
| **YOLO11** — [`yolo11_pipeline/`](yolo11_pipeline/) | The same pipeline with YOLO11n-seg, the model the team shipped in October. | Laptop, any GPU | [`yolo11_pipeline/INSTALL.md`](yolo11_pipeline/INSTALL.md) |

**Training vs fine-tuning.** The YOLO models are *trained* on the public WGISD photos (`train.py` in their folders) and shipped in `model/`. On our own vineyard clips every model is *fine-tuned* from its starting point: SAM 2.1 from Meta's pretrained weights (`fruit.py train`), YOLO26 and YOLO11 from their WGISD models (`fruit.py train-detector`, both in `sam2_pipeline/`), SAM 3 and SAM 3.1 from Meta's released weights (`finetune.py` in their folders); SAM 3 and SAM 3.1 are also scored *zero-shot*, as released. The leaderboard (`fruit.py evaluate`) scores all of them on the same held-out clips, and each row says which it is. All of this as HPCC batch jobs: [`hpcc/README.md`](hpcc/README.md).

## Layout

```
Capstone-MSU_Fruit/
├── data/              everything the pipelines read and write (data/README.md)
│   ├── wgisd/         public WGISD grape dataset (in git)
│   ├── wgisd_yolo/    WGISD as YOLO labels, with grape.yaml and the script that builds it (in git)
│   ├── videos/        the partner's video, synced from SharePoint (not in git)
│   ├── dataset/       our labelled clips (not in git)
│   ├── predictions/   every model's masks on the clips, what the leaderboard scores (not in git)
│   └── results/       every output: the leaderboard, then sam2/ sam3/ sam3_1/ yolo26/ yolo11/
├── sam2_pipeline/     code, install guide, environment, model/
├── sam3_pipeline/     code, guide, model/
├── sam3_1_pipeline/   the same scripts for SAM 3.1
├── yolo26_pipeline/   code, install guide, environment, model/
├── yolo11_pipeline/   the same scripts with the YOLO11 model
└── hpcc/              run_all.sh: fine-tune and score every model as HPCC batch jobs
```

Three rules: code lives in `<pipeline>/`; everything it reads or writes lives in `data/`; `<pipeline>/model/` holds the model that pipeline ships. [`data/README.md`](data/README.md) lists what every command writes and where.

## Data

Vineyard video (GoPro, phone, DJI drone) from National Grape, in the team's SharePoint library; it reaches the HPCC through Globus or `hpcc/sync_sharepoint.sh` ([`data/README.md`](data/README.md)). Public data: the Embrapa WGISD grape dataset (CC BY-NC 4.0), in `data/wgisd/`.

## Tools

Python · PyTorch · Segment Anything (SAM 2.1, SAM 3, SAM 3.1) · Ultralytics YOLO (YOLO26, YOLO11, YOLOE) · OpenCV · Jupyter / Google Colab · MSU HPCC
