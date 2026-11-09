# YOLO11 pipeline

YOLO11n-seg finds and outlines grape clusters in a photo or a video frame; it is the model the team shipped in October. This folder trains it, scores it, counts with it and runs it on video. It reads everything from `../data/` and writes training runs to `../data/runs/yolo11/` and everything else to `../data/results/yolo11/`. The model it ships is `model/best.pt`.

```
../data/wgisd_yolo ──▶ train.py ──▶ ../data/runs/yolo11/<run>/weights/best.pt ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

New here: [INSTALL.md](INSTALL.md). Every command runs from this folder with its environment active.

| File | What it does |
|---|---|
| `train.py` | Trains the model: on the public WGISD photos by default, on our own labels with `--data`. |
| `evaluate.py` | Box and mask mAP on the WGISD test photos; several weights make a table. |
| `count_test.py` | Cluster counts against hand counts on the 27 WGISD test photos. |
| `video.py` | Annotated video and per-frame counts for one video or a folder of videos; optional tracking. |
| `pipeline.py` | The names every script uses: the model file, the data folder, where runs and results go. |
| `model/` | The model this pipeline ships ([model/README.md](model/README.md)). |
| `hpcc/` | HPCC environment (`setup_env.sh`) and batch job (`run.sb`). |
| `tests/` | `python -m unittest discover -s tests` |
| `requirements.txt` | Python packages. |

The model detects grape clusters, not berries. Tracker ids are not verified unique clusters, and no count here is a yield estimate.

## 1. Train on WGISD

```bash
python train.py
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | `../data/wgisd_yolo/grape.yaml` | Dataset ([`../data/README.md`](../data/README.md)) |
| `--model` | `yolo11n-seg.pt` | Starting weights: an Ultralytics name (downloaded) or a file such as `model/best.pt` |
| `--epochs` | `50` | Epochs (stops early after 20 without improvement) |
| `--imgsz` | `512` | Image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--name` | `wgisd` | Run folder `../data/runs/yolo11/<name>/` |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

The run folder holds `weights/best.pt`, `results.csv`, `results.png`, the PR and F1 curves and the confusion matrix. 50 epochs take about 12 minutes on a 4 GB GPU and 80 minutes on a 2-core CPU.

A stronger WGISD model on a GPU: a bigger model, a larger image size, more epochs, evaluated at the training size.

```bash
python train.py --model yolo11s-seg.pt --imgsz 1024 --batch 8 --epochs 150 --name wgisd_s1024
python evaluate.py --model ../data/runs/yolo11/wgisd_s1024/weights/best.pt --imgsz 1024
```

Keep `imgsz` the same wherever a model is used (`video.py --imgsz`, its `imgsz:` in `../sam2_pipeline/configs/lineup.yaml`).

To ship a run: `cp ../data/runs/yolo11/<name>/weights/best.pt model/best.pt`, then update `model/README.md`.

## 2. Score it

```bash
python evaluate.py
python count_test.py
```

`evaluate.py` prints box and mask mAP@0.50 and mAP@0.50:0.95 on the 27 held-out test photos (plots in `../data/runs/yolo11/eval_test_<name>/`); `--model <weights>` scores another checkpoint, `--split val` the validation split, several `--model` weights write a side-by-side table to `../data/results/yolo11/evaluate_<split>.csv`. `count_test.py` compares cluster counts with hand counts on the same photos (`--conf`, default 0.25, trades missed clusters for false ones) and writes `../data/results/yolo11/count_test.csv` plus the annotated photos in `../data/results/yolo11/count_test/`.

These are scores on WGISD photos. The leaderboard in `../sam2_pipeline` (`python fruit.py evaluate`, in its own environment) scores this model on our own labelled clips next to every other model.

## 3. Run on video

```bash
python video.py --video ../data/videos/row3.mp4 --max-frames 0
```

Writes `../data/results/yolo11/row3_detect.mp4` (masks, boxes, confidences, a count panel) and `row3_detect.csv` (`frame`, `visible_now`, `tracked_now`, `tracker_ids_seen`).

| Argument | Default | Meaning |
| --- | --- | --- |
| `--video` / `--video-dir` | one required | One video, or every video under a folder at any depth, as a SharePoint sync leaves it (subfolders kept in the output; unreadable and still-syncing files skipped; finished ones skipped unless `--overwrite`) |
| `--model` | `model/best.pt` | Weights |
| `--mode` | `detect` | `detect`, or `track` (ByteTrack ids across frames) |
| `--conf` | `0.30` | Minimum confidence |
| `--imgsz` | `768` | Inference image size |
| `--max-frames` | `300` | Frames to process; `0` = the whole video (always the whole video with `--video-dir`) |
| `--start-sec` | `0` | Start time in seconds |
| `--output` | `../data/results/yolo11/<video>_<mode>.mp4` | Output video; the CSV gets the same name. A folder's videos go to `../data/results/yolo11/<folder>/` |

## 4. Fine-tune on our own video

Labels come from the SAM 2.1 labeller: label clips there, then export them as a YOLO dataset (`python fruit.py export-yolo`, from `../sam2_pipeline/`, writes `../data/yolo_dataset/`). Then start from the shipped model:

```bash
python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard
```

Labels made in Roboflow work the same way: export them as a YOLO segmentation dataset and point `--data` at its `data.yaml`.

On the HPCC the same fine-tuning is one job of the SAM 2.1 pipeline, scored by the leaderboard afterwards: `sbatch hpcc/run.sb train-detector --base ../yolo11_pipeline/model/best.pt` from `../sam2_pipeline/` ([`../hpcc/README.md`](../hpcc/README.md)).

## 5. HPCC

Once, on a development node (`ssh dev-amd20-v100`): `bash hpcc/setup_env.sh`. Then any script of this folder runs as a GPU job, submitted from this folder; the log is `yolo11-<jobid>.out` here.

```bash
sbatch hpcc/run.sb train.py --epochs 50 --batch 16
sbatch hpcc/run.sb evaluate.py
sbatch hpcc/run.sb video.py --video-dir ../data/videos/CodyKresta_1025
```

## 6. If something goes wrong

| Symptom | Fix |
| --- | --- |
| `Missing weights: …/model/best.pt` | Incomplete clone (`git status`), or train a model (section 1) and copy it there. |
| `Missing …/wgisd_yolo/grape.yaml` | `python ../data/wgisd_yolo/prepare_wgisd.py` |
| CUDA out of memory | Lower `--batch` (training) or `--imgsz` (video). |
| CUDA unavailable | The CPU is used. For a GPU, install a CUDA build of PyTorch (INSTALL.md). |
| Only 300 frames processed | Add `--max-frames 0`. |
| `tracked_now` stays 0 | Tracking needs `--mode track`; ByteTrack does not give every detection an id. |
| False positives or missed clusters on video | The model has only seen WGISD photos: fine-tune on our footage (section 4). |

Licences: [WGISD](https://github.com/thsant/wgisd) CC BY-NC 4.0, [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0. Repository: [`../LICENSE.txt`](../LICENSE.txt).
