# YOLO26 pipeline

YOLO26n-seg finds and outlines grape clusters in a photo or a video frame. This folder trains it, scores it and counts the clusters in a video with it. It reads everything from `../data/` and writes training runs to `../data/runs/yolo26/` and everything else to `../data/results/yolo26/`. The model it ships is `model/best.pt`.

```
../data/wgisd_yolo ──▶ train.py ──▶ ../data/runs/yolo26/<run>/weights/best.pt ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

New here: [INSTALL.md](INSTALL.md). Every command runs from this folder with its environment active.

| File | What it does |
|---|---|
| `train.py` | Trains the model: on the public WGISD photos by default, on our own labels with `--data`. |
| `evaluate.py` | Box and mask mAP on the WGISD test photos; several weights make a table. |
| `count_test.py` | Cluster counts against hand counts on the 27 WGISD test photos. |
| `video.py` | Counts the clusters in a video, or in every video of a folder: annotated video, per-frame counts, the total. |
| `pipeline.py` | The names every script uses (the model file, the data folder, where runs and results go) and `load()`, which opens a weights file and says what it is. |
| `model/` | The model this pipeline ships ([model/README.md](model/README.md)). |
| `hpcc/` | HPCC environment (`setup_env.sh`) and batch job (`run.sb`). |
| `tests/` | `python -m unittest discover -s tests` |
| `requirements.txt` | Python packages. |

The model detects grape clusters, not berries. A count is the number of clusters the tracker followed, not a hand count, and not a yield estimate.

## 1. Train on WGISD

```bash
python train.py
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | `../data/wgisd_yolo/grape.yaml` | Dataset ([`../data/README.md`](../data/README.md)) |
| `--model` | `yolo26n-seg.pt` | Starting weights: an Ultralytics name (downloaded) or a file such as `model/best.pt` |
| `--epochs` | `50` | Epochs (stops early after 20 without improvement) |
| `--imgsz` | `512` | Image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--name` | `wgisd` | Run folder `../data/runs/yolo26/<name>/` |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

The run folder holds `weights/best.pt`, `results.csv`, `results.png`, the PR and F1 curves and the confusion matrix. 50 epochs take about 12 minutes on a 4 GB GPU and 80 minutes on a 2-core CPU.

A stronger WGISD model on a GPU: a bigger model, a larger image size, more epochs.

```bash
python train.py --model yolo26s-seg.pt --imgsz 1024 --batch 8 --epochs 150 --name wgisd_s1024
python evaluate.py --model ../data/runs/yolo26/wgisd_s1024/weights/best.pt
```

To ship a run: `cp ../data/runs/yolo26/<name>/weights/best.pt model/best.pt`, then update `model/README.md`.

## 2. Score it

```bash
python evaluate.py
python count_test.py
```

`evaluate.py` prints box and mask mAP@0.50 and mAP@0.50:0.95 on the 27 held-out test photos (plots in `../data/runs/yolo26/eval_test_<name>/`); `--model <weights>` scores another checkpoint, `--split val` the validation split, several `--model` weights write a side-by-side table to `../data/results/yolo26/evaluate_<split>.csv`. `count_test.py` compares cluster counts with hand counts on the same photos (`--conf`, default 0.25, trades missed clusters for false ones) and writes `../data/results/yolo26/count_test.csv` plus the annotated photos in `../data/results/yolo26/count_test/`. Both score a model at the image size it was trained at.

These are scores on WGISD photos. The leaderboard in `../sam2_pipeline` (`python fruit.py evaluate`, in its own environment) scores this model on our own labelled clips next to every other model.

## 3. Count the clusters in a video

```bash
python video.py --video ../data/videos/row3.mp4
python video.py --video-dir ../data/videos/CodyKresta_1025
```

Every cluster gets an id that follows it from frame to frame (ByteTrack). The total is the number of ids followed for at least a second; shorter ones are mostly a cluster the tracker lost and found again. Writes `../data/results/yolo26/row3_track.mp4` (masks, boxes, ids, the count so far) and `row3_track.csv` (`frame`, `visible_now`, `tracked_now`, `clusters_so_far`), and adds the video's total to `../data/results/cluster_counts.csv`, the file every pipeline adds to.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--video` / `--video-dir` | one required | One video, or every video under a folder at any depth (subfolders kept in the output; unreadable and still-syncing files skipped; videos that have a total skipped unless `--overwrite`) |
| `--model` | `model/best.pt` | Weights |
| `--conf` | `0.30` | Minimum confidence |
| `--imgsz` | `auto` | The size a frame is shrunk to before the model sees it. `auto` tries every size on five frames of the video, prints what it finds at each and uses the one at which the model is most sure; a number fixes it |
| `--max-frames`, `--start-sec` | the whole video | A stretch of it, e.g. `--max-frames 300` for a quick look; only a whole video adds its total to the table |
| `--mode` | `track` | `detect` draws what is in each frame, without ids or a total |
| `--output` | `../data/results/yolo26/<video>_<mode>.mp4` | Output video; the CSV gets the same name. A folder's videos go to `../data/results/yolo26/<folder>/` |

The input size matters because a model trained on photos only sees clusters of about the size they had in those photos: this one, about 40 to 100 px tall in the image it is given. Footage shot closer needs a smaller size, wider shots a larger one.

## 4. Fine-tune on our own video

Labels come from the SAM 2.1 labeller: label clips there, then export them as a YOLO dataset (`python fruit.py export-yolo`, from `../sam2_pipeline/`, writes `../data/yolo_dataset/`). Then start from the shipped model:

```bash
python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard
```

Labels made in Roboflow work the same way: export them as a YOLO segmentation dataset and point `--data` at its `data.yaml`.

On the HPCC the same fine-tuning is one job of the SAM 2.1 pipeline, scored by the leaderboard afterwards: `sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt` from `../sam2_pipeline/` ([`../hpcc/README.md`](../hpcc/README.md)).

## 5. HPCC

Once, on a development node (`ssh dev-amd20-v100`): `bash hpcc/setup_env.sh`. Then any script of this folder runs as a GPU job, submitted from this folder; the log is `yolo26-<jobid>.out` here.

```bash
sbatch hpcc/run.sb train.py --epochs 50 --batch 16
sbatch hpcc/run.sb evaluate.py
sbatch hpcc/run.sb video.py --video-dir ../data/videos/CodyKresta_1025
```

[`../hpcc/count_rows.sh`](../hpcc/README.md) submits the last one for every pipeline at once.

## 6. If something goes wrong

| Symptom | Fix |
| --- | --- |
| `Missing weights: …/model/best.pt` | Incomplete clone (`git status`), or train a model (section 1) and copy it there. |
| `Missing …/wgisd_yolo/grape.yaml` | `python ../data/wgisd_yolo/prepare_wgisd.py` |
| CUDA out of memory | Lower `--batch` (training) or `--imgsz` (video). |
| CUDA unavailable | The CPU is used. For a GPU, install a CUDA build of PyTorch (INSTALL.md). |
| `… is a base model` | A base model such as `yolo26n-seg.pt` has no grape class. Use `model/best.pt` or a trained run. |
| `No clusters found` | The table printed at the start shows what the model saw at every input size. Nothing at any size, or false positives and missed clusters: the model has only seen WGISD photos, so label our footage ([`../LABELLING.md`](../LABELLING.md)) and fine-tune (section 4). |

Licences: [WGISD](https://github.com/thsant/wgisd) CC BY-NC 4.0, [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0. Repository: [`../LICENSE.txt`](../LICENSE.txt).
