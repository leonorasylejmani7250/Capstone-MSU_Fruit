# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

This repository contains our work for **CMSE 495 — Experiential Learning in Data Science** at Michigan State University.

The project focuses on developing a low-cost and scalable computer vision pipeline for estimating fruit yield from GoPro and mobile phone video. The system will use models such as **YOLO** and **Segment Anything Model (SAM)** to detect, segment, and count fruit in agricultural environments.

The project will initially focus on fruit detection and counting, with the goal of evaluating performance across different fruit types such as grapes, blueberries, apples, and peaches.

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
