# YOLO26 pipeline

YOLO26n-seg finds and outlines grape clusters in a photo or a video frame. This folder trains it, scores it and counts the clusters in a video with it. It reads everything from `../data/` and writes training runs to `../data/runs/yolo26/` and everything else to `../data/results/yolo26/`. The model it ships is `model/best.pt`.

```
../data/wgisd_yolo ──▶ train.py ──▶ ../data/runs/yolo26/<run>/weights/best.pt ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

On a laptop: [INSTALL.md](INSTALL.md).

## On the MSU HPCC

Set up once: [`../hpcc/README.md`](../hpcc/README.md), step 1. The SAM 2.1 environment it builds runs this folder too. Every block below can be pasted as it is into a terminal on a development node: log in, then `ssh dev-amd24-h200`.

### 1. Score the shipped model

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
source hpcc/activate.sh
python evaluate.py
python count_test.py
```

About a minute each. `evaluate.py` prints box and mask mAP on the 27 held-out WGISD test photos. `count_test.py` compares cluster counts with hand counts on the same photos and writes `../data/results/yolo26/count_test.csv` and the annotated photos in `../data/results/yolo26/count_test/`. These are scores on WGISD photos; the leaderboard of the SAM 2.1 pipeline scores this model on our own labelled clips.

### 2. Count the clusters in a video

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
source hpcc/activate.sh
python video.py --video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

Every cluster gets an id that follows it from frame to frame (ByteTrack). The total is the number of ids followed for at least a second. It writes `../data/results/yolo26/<video>_track.mp4` (masks, boxes, ids, the count so far) and `<video>_track.csv`, and adds the total to `../data/results/cluster_counts.csv`, the file every pipeline adds to. A total is the number of clusters the tracker followed, not a hand count, and not a yield estimate.

Every video of a folder, as a batch job (videos that have a total are skipped):

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
sbatch hpcc/run.sb video.py --video-dir ../data/videos/2026/Grape--DJIOsmo
```

### 3. Train on WGISD

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
source hpcc/activate.sh
python train.py
```

Ten minutes or less: 50 epochs on the public WGISD photos. It ends with `Best weights: …` and the path of the run folder under `../data/runs/yolo26/`, which also holds `results.csv`, the curves and the confusion matrix.

A stronger model (bigger network, larger images, more epochs), as a batch job:

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
sbatch hpcc/run.sb train.py --model yolo26s-seg.pt --imgsz 1024 --batch 8 --epochs 150 --name wgisd_s1024
```

When `squeue --me` no longer lists it, score that run:

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
source hpcc/activate.sh
python evaluate.py --model ../data/runs/yolo26/wgisd_s1024/weights/best.pt
```

To ship a run, copy its `weights/best.pt` over `model/best.pt` and update [model/README.md](model/README.md).

### 4. Fine-tune on our own clips

The labels live in the SAM 2.1 pipeline, so this is one of its jobs. It starts from the shipped model:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --name yolo26_grapes_v1
```

When `squeue --me` no longer lists it, the leaderboard scores it next to every other model:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

Count a video with it:

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
source hpcc/activate.sh
python video.py --video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4 --model ../data/runs/yolo26_grapes_v1/weights/best.pt
```

## Reference

### `train.py`

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | `../data/wgisd_yolo/grape.yaml` | Dataset ([`../data/README.md`](../data/README.md)) |
| `--model` | `yolo26n-seg.pt` | Starting weights: an Ultralytics name (downloaded) or a file such as `model/best.pt` |
| `--epochs` | `50` | Epochs (stops early after 20 without improvement) |
| `--imgsz` | `512` | Image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--name` | `wgisd` | Run folder `../data/runs/yolo26/<name>/`; a number is added when it exists |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

### `evaluate.py` and `count_test.py`

`evaluate.py`: `--model <weights>` scores another checkpoint, `--split val` the validation split; several `--model` weights write a side-by-side table to `../data/results/yolo26/evaluate_<split>.csv`; the plots are in `../data/runs/yolo26/eval_test_<name>/`. `count_test.py`: `--conf` (default 0.25) trades missed clusters for false ones. Both score a model at the image size it was trained at.

### `video.py`

| Argument | Default | Meaning |
| --- | --- | --- |
| `--video` / `--video-dir` | one required | One video, or every video under a folder at any depth (subfolders kept in the output; unreadable and still-syncing files skipped; videos that have a total skipped unless `--overwrite`) |
| `--model` | `model/best.pt` | Weights |
| `--conf` | `0.30` | Minimum confidence |
| `--imgsz` | `auto` | The size a frame is shrunk to before the model sees it. `auto` tries every size on five frames of the video, prints what it finds at each and uses the one at which the model is most sure; a number fixes it |
| `--max-frames`, `--start-sec` | the whole video | With `--video`: a stretch of it, e.g. `--max-frames 300` for a quick look. Only a whole video adds its total to the table; a folder is always counted whole |
| `--mode` | `track` | `detect` draws what is in each frame, without ids or a total |
| `--output` | `../data/results/yolo26/<video>_<mode>.mp4` | Output video; the CSV gets the same name. A folder's videos go to `../data/results/yolo26/<folder>/` |

The input size matters because a model trained on photos only sees clusters of about the size they had in those photos: this one, about 40 to 100 px tall in the image it is given. Footage shot closer needs a smaller size, wider shots a larger one.

### Files

| File | What it does |
|---|---|
| `train.py` | Trains the model: on the public WGISD photos by default, on our own labels with `--data`. |
| `evaluate.py` | Box and mask mAP on the WGISD test photos; several weights make a table. |
| `count_test.py` | Cluster counts against hand counts on the 27 WGISD test photos. |
| `video.py` | Counts the clusters in a video, or in every video of a folder: annotated video, per-frame counts, the total. |
| `pipeline.py` | The names every script uses (the model file, the data folder, where runs and results go) and `load()`, which opens a weights file and says what it is. |
| `model/` | The model this pipeline ships ([model/README.md](model/README.md)). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a batch job, `setup_env.sh` for an environment of its own. |
| `tests/` | `python -m unittest discover -s tests` |
| `requirements.txt` | Python packages. |

The model detects grape clusters, not berries. Outside the HPCC, fine-tuning on our own labels is two commands: `python fruit.py export-yolo` in `../sam2_pipeline/` writes `../data/yolo_dataset/`, then `python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard` here. Labels made in Roboflow work the same way: export them as a YOLO segmentation dataset and point `--data` at its `data.yaml`.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `You are currently using a gateway node`, or `module(s) could not be loaded` | You are on a gateway node: `ssh dev-amd24-h200`, then paste the block again. |
| `can't open file …/train.py`, or `No module named 'ultralytics'` | Paste the whole block: it starts with the `cd` and `source hpcc/activate.sh` lines this folder needs. |
| `Killed` in the middle of a run in the terminal | The process used up the two hours of CPU time it gets on a development node. Submit the run as a batch job: `sbatch hpcc/run.sb` and then the command without `python`. |
| A job log says a download failed | Fetch the base weights once on the development node: `source hpcc/activate.sh`, then `python -c "from ultralytics import YOLO; YOLO('yolo26s-seg.pt')"` in this folder, and submit again. |
| `Missing weights: …/model/best.pt` | Incomplete clone (`git status`), or train a model (step 3) and copy it there. |
| `Missing …/wgisd_yolo/grape.yaml` | `python ../data/wgisd_yolo/prepare_wgisd.py` |
| CUDA out of memory | Lower `--batch` (training) or `--imgsz` (video). |
| CUDA unavailable | The CPU is used. For a GPU, install a CUDA build of PyTorch (INSTALL.md). |
| `… is a base model` | A base model such as `yolo26n-seg.pt` has no grape class. Use `model/best.pt` or a trained run. |
| `No clusters found` | The table printed at the start shows what the model saw at every input size. Nothing at any size, or false positives and missed clusters: the model has only seen WGISD photos, so label our footage ([`../LABELLING.md`](../LABELLING.md)) and fine-tune (step 4). |

Licences: [WGISD](https://github.com/thsant/wgisd) CC BY-NC 4.0, [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0. Repository: [`../LICENSE.txt`](../LICENSE.txt).
