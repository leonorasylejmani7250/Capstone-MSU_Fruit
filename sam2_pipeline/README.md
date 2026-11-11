# SAM 2.1 pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb)

Label grape clusters in our video, fine-tune SAM 2.1 on the labels, score every model on the same held-out clips with the same rules, and count the clusters in a video.

```
videos ──▶ clips ──▶ labels ──▶ fine-tune ──▶ leaderboard ──▶ count the clusters in new video
```

On the MSU HPCC: the steps below. In Google Colab or on a laptop: [INSTALL.md](INSTALL.md), with a 10-minute demo.

## On the MSU HPCC

Set up once: [`../hpcc/README.md`](../hpcc/README.md), step 1. Every block below can be pasted as it is into a terminal on a development node: log in, then `ssh dev-amd24-h200`.

### 1. See what is in the project

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py status
```

It lists the videos, the clips cut from them, how many are labelled, the fine-tuned models and the last leaderboard.

### 2. Cut new video into clips, then label

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py clips
```

It cuts 16-frame clips from every video under `data/videos/` that has none yet. Then label them: [`../LABELLING.md`](../LABELLING.md).

### 3. Fine-tune on the labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v1 --steps 3000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --name yolo26_grapes_v1
```

Two batch jobs: SAM 2.1 from Meta's pretrained weights (→ `data/models/sam2.1_base_plus_grapes_v1.pt`) and the YOLO26 detector from its WGISD model (→ `data/runs/yolo26_grapes_v1/weights/best.pt`). `squeue --me` lists them until they are done; the logs are `fruit-<job id>.out` in this folder. A run name is trained once: after labelling more clips, use a new name such as `grapes_v2`.

### 4. The leaderboard

When `squeue --me` no longer lists the training jobs:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

A few minutes, in the terminal. It scores every model on the labelled val clips, prints two tables and saves them as `data/results/leaderboard_val.csv`:

- **Prompted** (SAM 2.1): shown each cluster once by a box or a click, must follow it. Headline **J&F** (0–100).
- **Automatic** (YOLO26, YOLO11, YOLOE, detector → SAM 2.1, SAM 3, SAM 3.1): gets only pixels. Headline **F1** at IoU 0.5, plus precision, recall, count error, tracks found vs true.

Row names say what each model is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<run>)`. SAM 3 and SAM 3.1 get their rows from step 2 of their own READMEs.

### 5. Count the clusters in a video

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb video videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

A detector finds the clusters and SAM 2.1 outlines and follows each one, so a cluster keeps one id for as long as it is in view; the total is the number of ids. It writes `data/results/sam2/overlay_<video>_0s.mp4`, `.counts.csv` (per frame) and `.tracks.csv` (one row per cluster), and adds the total to `data/results/cluster_counts.csv`, the file every pipeline adds to.

With the two models fine-tuned in step 3:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb video videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4 --detector runs/yolo26_grapes_v1/weights/best.pt --weights models/sam2.1_base_plus_grapes_v1.pt
```

Every video of a folder, as a batch job:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb video videos/2026/Grape--DJIOsmo
```

Every model's total for every video:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py counts
```

| Option of `video` | Default | Meaning |
|---|---|---|
| `--detector` | `yolo26_pipeline/model/best.pt` | The detector: any YOLO weights (`runs/<name>/weights/best.pt` for one fine-tuned on our clips), or `yoloe-26m-seg.pt`, which looks for `--text` |
| `--weights` | newest in `model/`, else pretrained | SAM 2.1 weights |
| `--conf` | `0.25` | Keep boxes that score at least this |
| `--detector-imgsz` | `auto` | The detector's input size (next step) |
| `--start`, `--seconds` | the whole video | A stretch of it, e.g. `--seconds 20` for a quick look; only a whole video adds its total to the table |

### 6. A model finds nothing in a video

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb check videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

It runs every detector on six frames of the video at every input size, prints a table (boxes per frame and top score), writes one picture per detector to `data/results/check/<video>/`, and says which detector and size to use, or that none works.

Three things make a detector silent:

| Cause | What `check` shows | Do this |
|---|---|---|
| The wrong weights: a base model such as `yolo26n-seg.pt`, which has no grape class, or a training run of an epoch or two | The weights line says so when the model loads | Use `yolo26_pipeline/model/best.pt` or a fine-tuned model |
| The wrong input size. A YOLO trained on photos only sees clusters about 40 to 100 px tall in the image it is given: footage shot closer needs a smaller size, wider shots a larger one | Boxes at some sizes, none at others | Nothing: the default size, `auto`, picks the size at which the detector is most sure |
| Footage unlike the photos the detector learned from, such as early-season video, where the clusters are flower clusters | No size has boxes, or a few low-scoring boxes sit on leaves | Label clips of this footage ([`../LABELLING.md`](../LABELLING.md)) and fine-tune |

### More about the jobs

`run.sb` asks for one A100 for 4 hours; override per job, for example `sbatch --gpus=v100:1 --time=02:00:00 hpcc/run.sb evaluate`. A training run that hits the time limit resumes when the same line is submitted again. Everything after `run.sb` goes to `fruit.py` (`python fruit.py --help` lists the commands); `bash --login hpcc/run.sb …` runs the same thing in the terminal instead of the queue. The project folder and the SAM 2.1 size are set in `hpcc/config.sh`.

## Reference

### Reading the results

- Fine-tuning is not guaranteed to help. Compare fine-tuned and pretrained on the same val clips, and watch the comparison video before trusting a gap of a point or two.
- Labels drawn with SAM 2.1's help flatter its *box* rows; lean on the click rows and the automatic table.
- A cluster total is the number of clusters a model followed. A cluster that is hidden and comes back is counted twice, a missed one not at all. The leaderboard's *tracks found / true* column is that check on the labelled clips; count one row by hand before relying on a video's total.
- A cluster count is not yield; converting counts to kilograms needs a calibration against hand-harvested vines.

### Scoring any model

The leaderboard scores every model in `configs/lineup.yaml` plus every model it finds in the project: SAM 2.1 weights in `model/` and `data/models/`, detectors from `train-detector` in `data/runs/`, SAM 3 and SAM 3.1 masks in `data/predictions/`. The table keeps the rows of earlier runs on the same clips, so models can be scored one at a time; `evaluate --fresh` starts it over.

To add a model, add an entry under `models:` in `configs/lineup.yaml`. Paths are relative to the project folder, then to the repository; an entry whose file is missing is skipped.

```yaml
  - {name: my YOLO-seg, type: ultralytics, weights: runs/yolo_detector/weights/best.pt}
  - {name: YOLO26 → SAM 2.1 fine-tuned, type: detect_then_sam2, checkpoint: sam2_pipeline/model/sam2.1_base_plus_grapes_v1.pt,
     detector: {type: ultralytics, weights: yolo26_pipeline/model/best.pt, conf: 0.25}}
  - {name: masks from another program, type: folder, path: predictions/my_masks, tracks: true}
  - {name: my model, type: python, class: ../examples/my_model.py:MyModel}
```

### Labels made elsewhere

Roboflow, CVAT, Label Studio or YOLO exports, with the path of your export in place of `export/…`:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py import-coco export/train/_annotations.coco.json
python fruit.py import-yolo export/train/images export/train/labels
python fruit.py approve --all-drafts
```

COCO or YOLO outlines are used as drawn. Box-only labels get a SAM 2.1 mask and stay drafts until checked in the labelling screen or approved with the third line. The split is by source video, read off the file names.

### Files

| File | What it is |
|---|---|
| `fruit.py` | The pipeline as a command line: `status`, `clips`, `train`, `train-detector`, `evaluate`, `video`, `counts`, `check`, … (`python fruit.py --help`). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a job, `setup_env.sh` builds the environment, `config.sh` holds the paths. |
| `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` | The whole pipeline as a notebook, with the labelling screen. Colab, or Jupyter on the HPCC. |
| `configs/lineup.yaml` | The models the leaderboard scores. Add any model here. |
| `model/` | The fine-tuned SAM 2.1 the team ships ([model/README.md](model/README.md)). |
| `INSTALL.md`, `environment.yml` | Install guide and conda environment for a laptop. |
| `demo.ipynb` | 10-minute demo on public grapes: score pretrained SAM 2.1, fine-tune it, score again. |
| `count_test.ipynb` | Our YOLO26 → SAM 2.1 counts against hand counts on 27 WGISD photos. |
| `tests/test_install.py` | One-minute install check. |
| `fruit_sam2.py` | The notebook's tool cells as one module. Generated; do not edit. |
| `source/` | The notebooks' cells and `build.py`. Edit a cell, run `python source/build.py`, commit the regenerated files. |
| `examples/my_model.py` | Template for scoring your own model. |

**The project folder** is [`../data/`](../data/README.md) (every `fruit.py` command uses it unless you pass `--project`): `videos/`, `dataset/` (clips and labels), `predictions/`, `models/`, `results/`, `runs/`. In Colab it is on Google Drive; the demo keeps to `data/demo/`.

### Words used here

| Word | Meaning |
|---|---|
| **clip** | 16 frames cut from a video, named after the video's path under `videos/`. Everything works on clips. |
| **label** | One mask per grape cluster per frame, drawn by SAM 2.1 from the box you draw. Stored in `data/dataset/Annotations/`. |
| **draft / done** | A clip's labels are a *draft* until you check them and press *Clip checked: mark done* (or run `fruit.py approve`). Only *done* clips are used for training and scoring. |
| **train / val** | Whole videos are held out as *val*. Models are fine-tuned on the train clips and scored on the val clips. |
| **pretrained** | SAM 2.1 as Meta released it. |
| **trained on WGISD** | A YOLO model trained on the public WGISD photos (the two YOLO pipelines). |
| **fine-tuned on our clips** | A model trained further on our labelled clips: `fruit.py train` for SAM 2.1 (from pretrained), `fruit.py train-detector` for YOLO (from the WGISD model). |
| **zero-shot** | No training: SAM 3, SAM 3.1 or YOLOE finding clusters from the text "grape cluster". |
| **prompted / automatic** | The two leaderboards: a model shown each cluster once (SAM 2.1, J&F score) vs a model that must find them itself (everything else, F1 score). |

## When something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded` | You are on a gateway node: `ssh dev-amd24-h200`, then paste the block again. |
| `conda: command not found`, `ModuleNotFoundError` | Paste the whole block: it starts with `source hpcc/activate.sh`, or runs through `hpcc/run.sb`. |
| `Unable to open file hpcc/run.sb`, or a job ends at once with an empty log | Submit from `sam2_pipeline/`: every block starts with the `cd` line. |
| `cannot find videos/…` | The path is under `data/videos/`: `python fruit.py status` lists every video with its path. |
| `Killed` in the middle of a run in the terminal | The process used up its two hours of CPU time on the development node. Submit the run as a batch job: the same line with `sbatch` in place of `bash --login`. |
| Job log: weights download fails | Compute nodes may be offline. On a development node: `python fruit.py download --sizes base_plus`. |
| `CUDA out of memory` while scoring | Smaller SAM 2.1 (`size:` in `configs/lineup.yaml`) or a bigger GPU (`sbatch --gpus=a100:1 …`). |
| `setup_env.sh` fails at `pip install torch` | Run it on a development node. If its final check fails on a GPU node, ask ICER which CUDA version the drivers support, `pip uninstall -y torch torchvision`, change `cu126` in the script, re-run. |
| Labelling screen blank in Colab | Runtime ▸ Restart session, Run all. In OnDemand, use JupyterLab. |
| Fine-tuned model scores worse | Fewer steps, or more labelled clips; check the labels the comparison video shows going wrong. |

Licences: SAM 2 code and weights Apache 2.0; Ultralytics YOLO/YOLOE AGPL-3.0; demo images WGISD, CC BY-NC 4.0.
