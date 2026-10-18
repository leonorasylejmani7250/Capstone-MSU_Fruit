# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

This repository contains our work for **CMSE 495 — Experiential Learning in Data Science** at Michigan State University.

The project focuses on developing a low-cost and scalable computer vision pipeline for estimating fruit yield from GoPro and mobile phone video. The system will use models such as **YOLO** and **Segment Anything Model (SAM)** to detect, segment, and count fruit in agricultural environments.

The project will initially focus on fruit detection and counting, with the goal of evaluating performance across different fruit types such as grapes, blueberries, apples, and peaches.

## Start here

There are two pipelines, one built on SAM 2.1 and one on YOLO11. They are separate: each has its own guide and its own Python environment. Follow the guide for the one you want.

| Pipeline | What it does | Guide |
|---|---|---|
| **SAM 2.1** | Label grape clusters in our videos with SAM 2.1's help, fine-tune SAM 2.1, score it and any other model on the same held-out clips, and run it on new video. Runs on a laptop, in Google Colab and on the MSU HPCC. | [INSTALL.md](INSTALL.md) to install it and run its demo, then [sam2_pipeline/README.md](sam2_pipeline/README.md) for everything else. |
| **YOLO11** | Train YOLO11n-seg on the public WGISD grape dataset, evaluate it on WGISD's test images, and run detection and tracking on video. | [INSTALL_Moe_YOLO.md](INSTALL_Moe_YOLO.md) to install it and run its demo (macOS, Linux and Windows), then [instructions.md](instructions.md) for training, tracking and the other options (its commands are written for Windows PowerShell). |

## Quick start: the SAM 2.1 demo

**Full steps: [INSTALL.md](INSTALL.md).** In short:

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit
conda env create --prefix ./envs --file environment.yml && conda activate ./envs
python tests/test_install.py
jupyter notebook
```

The test ends with `INSTALL OK`. In Jupyter, open `demo.ipynb` and choose *Run ▸ Run All Cells*. It takes 5–10 minutes on a laptop CPU.

No install: open [demo.ipynb in Google Colab](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/demo.ipynb).

## What is in this repository

| Path | What it is |
|---|---|
| `demo.ipynb` | Short end-to-end demo on public sample grapes: score pretrained SAM 2.1, fine-tune it, score again, compare. |
| `INSTALL.md`, `environment.yml` | Install instructions and the conda environment for the SAM 2.1 pipeline. |
| `tests/test_install.py` | One-minute check that the SAM 2.1 install works. |
| `sam2_pipeline/` | The SAM 2.1 pipeline: upload video, label grape clusters, fine-tune SAM 2.1, score any model, run on new video. Runs in Google Colab and on the MSU HPCC. Start with [`sam2_pipeline/README.md`](sam2_pipeline/README.md). |
| `data/demo_project/`, `data/msu_fruit_project/` | Local SAM 2.1 data and results, not in git. The demo writes to `data/demo_project/`, our videos go in `data/msu_fruit_project/videos/`. |
| `INSTALL_Moe_YOLO.md`, `instructions.md`, `requirements.txt` | The YOLO11 pipeline's install guide, its full guide (training, evaluation, detection and tracking on video) and its packages. |
| `prepare_wgisd.py`, `train.py`, `evaluate.py`, `video.py`, `preview_labels.py`, `extract_dji_frames.py`, `test_yolo.py` | The YOLO11 scripts. `grape.yaml` is their dataset file. |
| `external/wgisd/`, `data/wgisd_seg/` | The public WGISD grape dataset and its converted YOLO labels, both committed. |
| `runs/`, `outputs/` | Trained YOLO11 checkpoints (`runs/yolo11_wgisd2/weights/best.pt`) with their training logs, per-frame detection CSVs and label previews. The annotated result videos (9.5 GB, Git LFS) are kept on the `Moe` branch only. |
| `tests/test_smoke.py` | YOLO11 conversion and rendering tests: `python -m unittest discover -s tests -v`. |

## Main Goals

- Preprocess image and video data
- Detect fruit and fruit clusters
- Segment detected fruit
- Count fruit from images and video
- Evaluate model performance
- Compare performance across fruit types and camera systems
- Create visualizations of the results

## Tools

- Python
- YOLO
- Segment Anything Model (SAM)
- Computer vision and image-processing libraries

## Data

The project uses image and video data collected from vineyards and orchards, including GoPro and mobile phone footage.

## Course

**CMSE 495 — Experiential Learning in Data Science**  
Michigan State University

## YOLO11 Implementation

This project implements YOLO11n-seg for grape cluster detection and instance segmentation using WGISD.

### Features
- Dataset preparation and label visualization
- YOLO11n-seg training and evaluation
- Video detection with bounding boxes, masks and confidence scores
- Optional ByteTrack tracking
- Video and CSV output

### Training
Run prepare_wgisd.py before training.
Command: python train.py --epochs 50 --imgsz 512 --batch 2

### Video Inference
Run video.py with --video, --model, --mode detect and --max-frames 0.

The dataset, video files, weights and generated outputs are stored locally.
