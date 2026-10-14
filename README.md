# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

This repository contains our work for **CMSE 495 — Experiential Learning in Data Science** at Michigan State University.

The project focuses on developing a low-cost and scalable computer vision pipeline for estimating fruit yield from GoPro and mobile phone video. The system will use models such as **YOLO** and **Segment Anything Model (SAM)** to detect, segment, and count fruit in agricultural environments.

The project will initially focus on fruit detection and counting, with the goal of evaluating performance across different fruit types such as grapes, blueberries, apples, and peaches.

## Getting started

**Install and run the demo: see [INSTALL.md](INSTALL.md).** In short:

```bash
git clone https://github.com/moehocbaidi/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit
conda env create --prefix ./envs --file environment.yml && conda activate ./envs
python tests/test_install.py          # prints INSTALL OK
jupyter notebook                      # open demo.ipynb, Run All (5–10 min on a laptop CPU)
```

No install: open [demo.ipynb in Google Colab](https://colab.research.google.com/github/moehocbaidi/Capstone-MSU_Fruit/blob/main/demo.ipynb).

## What is in this repository

| Path | What it is |
|---|---|
| `demo.ipynb` | Short end-to-end demo on public sample grapes: score pretrained SAM 2.1, fine-tune it, score again, compare. |
| `INSTALL.md`, `environment.yml` | Install instructions and the conda environment. |
| `tests/test_install.py` | One-minute check that the install works. |
| `sam2_pipeline/` | The SAM 2.1 pipeline: upload video, label grape clusters, fine-tune SAM 2.1, score any model, run on new video. Runs in Google Colab and on the MSU HPCC. Start with [`sam2_pipeline/README.md`](sam2_pipeline/README.md). |
| `data/` | Local data and results, not in git. The demo writes to `data/demo_project/`, our videos go in `data/msu_fruit_project/videos/`. |

The YOLO11 grape segmentation work is on the `Moe` branch.

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
