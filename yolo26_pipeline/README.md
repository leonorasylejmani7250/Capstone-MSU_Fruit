# MSU FRUIT · YOLO26 pipeline

Train **YOLO26n-seg** (Ultralytics, 2026) to find and outline grape clusters, evaluate it, count clusters on images, and run it on vineyard video with optional tracking. It trains on the public **WGISD** grape dataset in `../data/wgisd` and ships its model in `model/best.pt`.

```
data/wgisd ──▶ prepare_wgisd.py ──▶ data/wgisd_yolo ──▶ train.py ──▶ data/runs/yolo26/<run> ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

**New here? [INSTALL.md](INSTALL.md)** installs the pipeline (macOS, Linux, Windows) and runs the demo on public images. Everything below runs from this folder, `yolo26_pipeline/`, with its environment active.

| File | What it is |
|---|---|
| `INSTALL.md`, `requirements.txt` | Install guide and the Python packages. |
| `grape.yaml` | The dataset for training and evaluation: `../data/wgisd_yolo`. |
| `prepare_wgisd.py` | Converts WGISD's masks into YOLO polygon labels (`../data/wgisd_yolo`; already done and committed). |
| `preview_labels.py` | Draws the labels on a few images, to check them (`../data/results/yolo26/label_preview/`). |
| `train.py` | Trains YOLO26n-seg; runs go to `../data/runs/yolo26/<name>/`. |
| `evaluate.py` | Box and mask mAP on the held-out WGISD test split. |
| `count_test.py` | Cluster counts vs hand counts on the 27 test images (`../data/results/yolo26/count_test*`). |
| `video.py` | Annotated video and per-frame counts, with or without ByteTrack tracking (`../data/results/yolo26/`). |
| `extract_frames.py` | Still frames from a video, every few seconds, for labelling (`../data/frames_to_label/`). |
| `tests/test_smoke.py` | No-GPU unit tests of the conversion and the rendering. |
| `hpcc/` | `setup_env.sh` (one-time environment on the MSU HPCC) and `run.sb` (any script here as a GPU batch job). |
| `model/` | The model this pipeline ships, `best.pt`, with a README on how it was trained and how it scored. |

> **The model detects grape clusters, not berries.** `visible_now` counts detections in one frame. Tracker ids are not verified unique clusters and no count here is a yield estimate: repeated views, missed detections and id switches all add error.

## 1. The data

The public [WGISD](https://github.com/thsant/wgisd) dataset (300 photos, 5 varieties, hand-drawn cluster masks, CC BY-NC 4.0) lives in `../data/wgisd/`, shared with the SAM 2.1 demo. `prepare_wgisd.py` turns its masks into YOLO polygon labels in `../data/wgisd_yolo/` (`images/` and `labels/`, each with `train/` 94, `val/` 16, `test/` 27 images; WGISD's own train/test lists, with 15% of train held out as val, seed 42). The converted labels are committed, so this step is optional:

```bash
python prepare_wgisd.py
python preview_labels.py --split val --limit 8
```

A different OpenCV version may trace a few outlines slightly differently, so `git status` can show small changes under `data/wgisd_yolo/labels` afterwards; `git checkout -- ../data/wgisd_yolo` restores the committed labels. The preview images in `../data/results/yolo26/label_preview/` should show yellow outlines on the grape clusters.

## 2. Train

```bash
python train.py --epochs 50 --imgsz 512 --batch 2
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `50` | Maximum number of training epochs (early stopping after 20 without improvement) |
| `--imgsz` | `512` | Training image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--data` | `grape.yaml` | Dataset YAML; `../data/yolo_dataset/data.yaml` for our own labels (section 5) |
| `--model` | `yolo26n-seg.pt` | Starting weights: a bare Ultralytics name (downloaded; `yolo26s-seg.pt` is the next size up) or a file such as `model/best.pt` |
| `--name` | `wgisd` | Run name: output goes to `../data/runs/yolo26/wgisd/` (a number is added if it exists) |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

The run folder holds `weights/best.pt` and `last.pt`, `results.csv` and `results.png` (losses and mAP per epoch), the PR and F1 curves, and the confusion matrix. The October YOLO11 run took 12 minutes on a 4 GB NVIDIA GPU; the YOLO26 run in `model/` took 80 minutes on a 2-core CPU. A one-epoch run (`--epochs 1 --name smoke`) checks the install in under two minutes on a CPU.

**On the MSU HPCC** (a V100 trains the 50 epochs in minutes): once, on a development node, `cd ~/Capstone-MSU_Fruit/yolo26_pipeline && bash hpcc/setup_env.sh`; then `sbatch hpcc/run.sb train.py --epochs 50 --imgsz 512 --batch 16`. `squeue --me` shows the job, `yolo26-<jobid>.out` is its log, and the run lands in `../data/runs/yolo26/` like a local one. `sbatch hpcc/run.sb evaluate.py` and `sbatch hpcc/run.sb video.py …` work the same way. (The HPCC scripts follow ICER's documentation and the SAM 2.1 pipeline's scripts, but have not been run on the HPCC yet.)

**Training for accuracy.** The shipped model was trained on a CPU with the smallest settings that finish there (`yolo26n`, 512 px, batch 2). WGISD photos are 2048 px wide and many clusters are small, so on a GPU the first things to try are a larger input size, a larger model and more epochs, then compare on the test split with `evaluate.py` and on our own clips with the leaderboard:

```bash
sbatch hpcc/run.sb train.py --model yolo26s-seg.pt --imgsz 1024 --batch 8 --epochs 150 --name wgisd_s1024
sbatch hpcc/run.sb evaluate.py --model ../data/runs/yolo26/wgisd_s1024/weights/best.pt --imgsz 1024
```

Evaluate at the size you trained at, and keep `imgsz` consistent wherever the model is used (`video.py --imgsz`, the `imgsz:` of its lineup entry in `../sam2_pipeline/configs/lineup.yaml`). What matters more than any of this for the project is training on our own vineyard footage: section 5. None of these larger settings has been run yet; the numbers in `model/README.md` are the baseline to beat.

**When a run is the one to use, copy its weights into `model/`** and update `model/README.md`: `cp ../data/runs/yolo26/wgisd/weights/best.pt model/best.pt`. Every other script uses `model/best.pt` by default.

## 3. Evaluate

On the held-out WGISD test split (27 images that were never trained on):

```bash
python evaluate.py
```

It prints box and mask **mAP@0.50** and **mAP@0.50:0.95** and saves Ultralytics' plots to `../data/runs/yolo26/eval_test/`. `--model <weights>` scores another checkpoint, `--split val` the validation split. These are scores on WGISD photos, not on vineyard video; the leaderboard in `../sam2_pipeline` scores this model on our own labelled clips next to SAM 2.1 and SAM 3.1 (`python ../sam2_pipeline/fruit.py evaluate`).

**Count test** (Michal's): the model's cluster count against hand counts on the same 27 images:

```bash
python count_test.py
```

It prints the error per image and the mean absolute error, and writes `../data/results/yolo26/count_test.csv` plus the annotated images in `../data/results/yolo26/count_test/`. `--conf` (default 0.25) trades missed clusters for false ones.

## 4. Run on a video

Put the video in `../data/videos/` (not in git; see `../data/README.md`).

```bash
python video.py --video ../data/videos/row3.mp4 --max-frames 0
```

Writes `../data/results/yolo26/row3_detect.mp4` (masks, boxes, confidences and a count panel on every frame) and `row3_detect.csv` (one row per frame). `--max-frames 0` means the whole video; the default stops after 300 frames, which is handy for a first look.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--video` | required | Input video |
| `--model` | `model/best.pt` | Weights |
| `--output` | `../data/results/yolo26/<video>_<mode>.mp4` | Annotated video; the CSV gets the same name |
| `--mode` | `detect` | `detect`, or `track` (ByteTrack ids across frames) |
| `--conf` | `0.30` | Minimum confidence |
| `--imgsz` | `768` | Inference image size |
| `--max-frames` | `300` | Frames to process; `0` = the whole video |
| `--start-sec` | `0` | Where to start, in seconds |
| `--tracker` | `bytetrack.yaml` | Tracker settings for `--mode track` |

CSV columns: `frame`, `visible_now` (clusters detected in this frame), `tracked_now` (detections that got a tracker id), `tracker_ids_seen` (distinct ids so far; not a cluster count).

**Every video in a folder** (bash; the same loop in PowerShell is in INSTALL.md):

```bash
for v in ../data/videos/*.mp4; do python video.py --video "$v" --max-frames 0; done
```

## 5. Frames for labelling

```bash
python extract_frames.py --video ../data/videos/row3.mp4 --every-sec 5 --max-images 60
```

Saves still frames to `../data/frames_to_label/row3/`. Label grape *clusters*, and keep some labelled frames of the same session aside for testing. The SAM 2.1 pipeline's labeller (`../sam2_pipeline`) is the faster way to label clips, and `python ../sam2_pipeline/fruit.py export-yolo` writes its labels as a YOLO dataset in `../data/yolo_dataset/`. Fine-tune the shipped model on them:

```bash
python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard
```

The same thing as one batch job on the HPCC, scored automatically by the leaderboard afterwards: `python ../sam2_pipeline/fruit.py train-detector --base ../yolo26_pipeline/model/best.pt` (`../hpcc/README.md`).

## 6. What was tested

On a 2-core Linux CPU with Python 3.12, PyTorch 2.14.1 and Ultralytics 8.4.198 (November 2026, this layout): the unit tests, `prepare_wgisd.py`, `preview_labels.py`, a one-epoch `train.py` and the 50-epoch run that made `model/best.pt`, `evaluate.py`, `count_test.py` (with a confidence sweep), `video.py` in both modes on the five-frame public demo video from INSTALL.md, and the model's entry in the SAM 2.1 leaderboard. The unit tests and the install check were also run from a fresh clone. The October YOLO11 run (`../data/results/yolo11/`) was trained on Windows with a 4 GB NVIDIA GPU, Python 3.11 and Ultralytics 8.3.70, and evaluated again on Linux. macOS and Windows have not been run with this layout yet.

## 7. If something goes wrong

| Symptom | Fix |
| --- | --- |
| `Missing weights: …/model/best.pt` | The model was not cloned or not trained yet: `git status` in `model/`, or train one (section 2) and copy it there. |
| `Missing grape.yaml` | Run `prepare_wgisd.py`, which rewrites it. |
| CUDA out of memory | Lower `--batch` for training or `--imgsz` for video. |
| CUDA unavailable | The CPU is used; fine for the demo, slow for training and long videos. Install a CUDA build of PyTorch for a GPU. |
| Only 300 frames processed | Add `--max-frames 0`. |
| `tracked_now` stays 0 | Tracking needs `--mode track`, and ByteTrack does not assign an id to every detection. |
| False positives or missed clusters on video | The model has only seen WGISD photos. Label frames from the vineyard video and fine-tune (section 5). |

Dataset and tools: [WGISD](https://github.com/thsant/wgisd) (CC BY-NC 4.0), [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [PyTorch](https://pytorch.org/). Repository licence: [`../LICENSE.txt`](../LICENSE.txt).
