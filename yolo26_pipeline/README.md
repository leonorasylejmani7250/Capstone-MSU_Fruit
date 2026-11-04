# YOLO26 pipeline

Train YOLO26n-seg to find and outline grape clusters, evaluate it, count clusters, and run it on video with optional tracking. It trains on the public WGISD dataset in `../data/wgisd` and ships its model in `model/best.pt`.

```
data/wgisd ──▶ prepare_wgisd.py ──▶ data/wgisd_yolo ──▶ train.py ──▶ data/runs/yolo26/<run> ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

New here: [INSTALL.md](INSTALL.md). Every command below runs from this folder with its environment active.

| File | What it is |
|---|---|
| `INSTALL.md`, `requirements.txt` | Install guide and Python packages. |
| `grape.yaml` | The dataset for training and evaluation (`../data/wgisd_yolo`). |
| `prepare_wgisd.py` | WGISD masks → YOLO labels in `../data/wgisd_yolo` (already done and committed). |
| `preview_labels.py` | Draws the labels on a few images (`../data/results/yolo26/label_preview/`). |
| `train.py` | Trains YOLO26n-seg; runs go to `../data/runs/yolo26/<name>/`. |
| `evaluate.py` | Box and mask mAP on the WGISD test split; several models side by side. |
| `count_test.py` | Cluster counts vs hand counts on the 27 test images. |
| `video.py` | Annotated video and per-frame counts, one video or a folder, optional ByteTrack tracking. |
| `extract_frames.py` | Still frames from a video for labelling (`../data/frames_to_label/`). |
| `tests/test_smoke.py` | Unit tests (no GPU). |
| `hpcc/` | HPCC environment (`setup_env.sh`) and batch job (`run.sb`). |
| `model/` | The model this pipeline ships ([model/README.md](model/README.md)). |

The model detects grape clusters, not berries. Tracker ids are not verified unique clusters, and no count here is a yield estimate.

## 1. The data

WGISD (300 photos, hand-drawn cluster masks, CC BY-NC 4.0) is in `../data/wgisd/`; its YOLO labels are in `../data/wgisd_yolo/` (train 94, val 16, test 27). To rebuild the labels and check them:

```bash
python prepare_wgisd.py
python preview_labels.py --split val --limit 8
```

A different OpenCV version moves a few outline vertices; `git checkout -- ../data/wgisd_yolo` restores the committed labels.

## 2. Train

```bash
python train.py --epochs 50 --imgsz 512 --batch 2
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `50` | Epochs (early stop after 20 without improvement) |
| `--imgsz` | `512` | Image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--data` | `grape.yaml` | Dataset YAML; `../data/yolo_dataset/data.yaml` for our own labels (section 5) |
| `--model` | `yolo26n-seg.pt` | Starting weights: an Ultralytics name (downloaded) or a file such as `model/best.pt` |
| `--name` | `wgisd` | Run folder `../data/runs/yolo26/<name>/` |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

The run folder holds `weights/best.pt`, `results.csv`, `results.png`, the PR and F1 curves and the confusion matrix. 50 epochs take about 12 minutes on a 4 GB GPU and 80 minutes on a 2-core CPU.

On the HPCC: once, `bash hpcc/setup_env.sh` on a development node; then `sbatch hpcc/run.sb train.py --epochs 50 --imgsz 512 --batch 16` (log: `yolo26-<jobid>.out`). `evaluate.py` and `video.py` run the same way.

For a more accurate WGISD model on a GPU, try a larger input size, a larger model and more epochs, and evaluate at the training size:

```bash
sbatch hpcc/run.sb train.py --model yolo26s-seg.pt --imgsz 1024 --batch 8 --epochs 150 --name wgisd_s1024
sbatch hpcc/run.sb evaluate.py --model ../data/runs/yolo26/wgisd_s1024/weights/best.pt --imgsz 1024
```

Keep `imgsz` the same wherever the model is used (`video.py --imgsz`, its `imgsz:` in `../sam2_pipeline/configs/lineup.yaml`). What matters most is training on our own vineyard footage: section 5.

To ship a run: `cp ../data/runs/yolo26/<name>/weights/best.pt model/best.pt` and update `model/README.md`.

## 3. Evaluate

```bash
python evaluate.py
```

Prints box and mask mAP@0.50 and mAP@0.50:0.95 on the 27 held-out test images; plots in `../data/runs/yolo26/eval_test_<model>/`. `--model <weights>` scores another checkpoint, `--split val` the validation split. Several models make a table, `../data/results/yolo26/evaluate_<split>.csv`:

```bash
python evaluate.py --model model/best.pt ../yolo11_pipeline/model/best.pt
```

These are scores on WGISD photos. The leaderboard in `../sam2_pipeline` (`python fruit.py evaluate`, in its own environment) scores this model on our own labelled clips next to SAM 2.1 and SAM 3.1.

Count test against hand counts on the same 27 images (`--conf`, default 0.25, trades missed clusters for false ones):

```bash
python count_test.py
```

Writes `../data/results/yolo26/count_test.csv` and the annotated images in `../data/results/yolo26/count_test/`.

## 4. Run on a video

```bash
python video.py --video ../data/videos/row3.mp4 --max-frames 0
```

Writes `../data/results/yolo26/row3_detect.mp4` (masks, boxes, confidences, a count panel) and `row3_detect.csv` (`frame`, `visible_now`, `tracked_now`, `tracker_ids_seen`). `--max-frames 0` means the whole video; the default stops after 300 frames.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--video` / `--video-dir` | one required | A video, or a folder of videos |
| `--model` | `model/best.pt` | Weights |
| `--output` | `../data/results/yolo26/<video>_<mode>.mp4` | Annotated video; the CSV gets the same name |
| `--mode` | `detect` | `detect`, or `track` (ByteTrack ids across frames) |
| `--conf` | `0.30` | Minimum confidence |
| `--imgsz` | `768` | Inference image size |
| `--max-frames` | `300` | Frames to process; `0` = whole video |
| `--start-sec` | `0` | Start time in seconds |
| `--tracker` | `bytetrack.yaml` | Tracker settings for `--mode track` |

Every video in a folder (whole videos; finished ones are skipped, `--overwrite` redoes them):

```bash
python video.py --video-dir ../data/videos/CodyKresta_1025
```

Outputs: `../data/results/yolo26/CodyKresta_1025/<video>_detect.mp4` and `.csv`.

## 5. Fine-tune on our own video

Label clips with the SAM 2.1 labeller (`../sam2_pipeline/README.md`) and export them as a YOLO dataset (`python fruit.py export-yolo`, from `sam2_pipeline/`), then:

```bash
python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard
```

On the HPCC the same thing is one job in the SAM 2.1 pipeline, scored by the leaderboard afterwards: `sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt` (`../hpcc/README.md`).

Still frames for labelling by hand: `python extract_frames.py --video ../data/videos/row3.mp4 --every-sec 5 --max-images 60` → `../data/frames_to_label/row3/`.

## 6. If something goes wrong

| Symptom | Fix |
| --- | --- |
| `Missing weights: …/model/best.pt` | Train a model (section 2) and copy it there. |
| `Missing grape.yaml` | `python prepare_wgisd.py` |
| CUDA out of memory | Lower `--batch` (training) or `--imgsz` (video). |
| CUDA unavailable | The CPU is used. For a GPU, install a CUDA build of PyTorch. |
| Only 300 frames processed | Add `--max-frames 0`. |
| `tracked_now` stays 0 | Tracking needs `--mode track`; ByteTrack does not give every detection an id. |
| False positives or missed clusters on video | The model has only seen WGISD photos: fine-tune on our footage (section 5). |

Licences: [WGISD](https://github.com/thsant/wgisd) CC BY-NC 4.0, [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0. Repository: [`../LICENSE.txt`](../LICENSE.txt).
