# MSU FRUIT: HPCC instructions, labelling guide and pipeline guide
The guides on `main` of [Capstone-MSU_Fruit](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit) (commit `2487232`) as one file, in this order:

1. [HPCC instructions](#hpcc-instructions): the whole workflow on the MSU HPCC, from logging in to the cluster counts
2. [Labelling guide](#labelling-guide)
3. The pipelines, one at a time: [SAM 2.1](#sam-21-pipeline), [SAM 3](#sam-3-pipeline), [SAM 3.1](#sam-31-pipeline), [YOLO26](#yolo26-pipeline), [YOLO11](#yolo11-pipeline)

A link from one of these guides to another jumps inside this file. Links to the guides that are not in it (`data/README.md`, the laptop `INSTALL.md` guides, the `model/` READMEs) open on GitHub.

---

## HPCC instructions

*From `hpcc/README.md`.*

Five steps. Every block can be pasted into a terminal as it is.

### Log in

Use your NetID in place of `netid`. Paste the two lines one at a time: `ssh` swallows whatever is pasted after it.

```bash
ssh netid@hpcc.msu.edu
```

```bash
ssh dev-amd24-h200
```

The first line lands on a gateway node, which runs no software and refuses `sbatch`. The second goes to a development node, where everything below is typed: the prompt then shows `dev-amd24-h200`.

### 1. Set up, once per person

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit
bash hpcc/setup.sh
```

It does two things:

- **Shared data**: `data/videos` and `data/dataset` become links to `/mnt/research/MSU-Fruit/videos` and `/mnt/research/MSU-Fruit/dataset`, so the team has one copy of the video and one set of clips and labels. Videos or clips already in the clone move there first. Everything else in `data/` stays in the clone: each person's own runs and results.
- **Environments**: SAM 2.1, which also runs the YOLO pipelines, then SAM 3 and SAM 3.1.

**Meta's weights for SAM 3 and SAM 3.1.** Request access at [facebook/sam3](https://huggingface.co/facebook/sam3) and [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) with one Hugging Face account and wait for the two approval emails. Then log in, once. It asks for a token: make one at <https://huggingface.co/settings/tokens> and paste it.

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
hf auth login
```

Then download the two files, 3 to 4 GB each:

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
hf download facebook/sam3 sam3.pt --local-dir model
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir ../sam3_1_pipeline/model
```

Later, to get the team's latest code:

```bash
cd ~/Capstone-MSU_Fruit
git pull
```

### 2. Bring in the video

[`data/README.md`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/data/README.md), "Video from SharePoint".

### 3. Label clips

[`LABELLING.md`](#labelling-guide). Labels made in Roboflow or CVAT come in with `fruit.py import-coco` / `import-yolo` ([`sam2_pipeline/README.md`](#sam-21-pipeline), "Labels made elsewhere").

### 4. Fine-tune every model and score them all

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1
```

It submits a chain of batch jobs and returns at once. Every model learns from the labelled `train` clips. The last job scores all of them on the `val` clips, next to the models that were not fine-tuned: `data/results/leaderboard_val.csv`. Each row says what the model is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<name>)`.

| Model | Starts from | Fine-tuned model |
|---|---|---|
| SAM 2.1 | Meta's pretrained `base_plus` | `data/models/sam2.1_base_plus_<name>.pt` |
| YOLO26 | `yolo26_pipeline/model/best.pt`, trained on WGISD | `data/runs/yolo26_<name>/weights/best.pt` |
| YOLO11 | `yolo11_pipeline/model/best.pt`, trained on WGISD | `data/runs/yolo11_<name>/weights/best.pt` |
| SAM 3 | Meta's release, also scored as released | `sam3_pipeline/model/sam3_<name>.pt` |
| SAM 3.1 | Meta's release, also scored as released | `sam3_1_pipeline/model/sam3_1_<name>.pt` |

A run name is trained once. After labelling more clips, use a new name, and give more steps and epochs if you want them (SAM 2.1 steps, YOLO epochs, SAM 3 epochs; the defaults are 3000, 100 and 20):

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v2 5000 150 30
```

**Ship a winner.** Copy it into its pipeline's `model/` folder (a YOLO model over `model/best.pt`) and fill in that folder's README. Step 5 counts with the models in the `model/` folders.

### 5. Count the clusters in every video

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/count_rows.sh
```

One folder under `data/videos`, or one video:

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/count_rows.sh 2026/Grape--DJIOsmo
```

One job per pipeline, each with the model it ships:

| Pipeline | How it counts |
|---|---|
| SAM 2.1 | YOLO26 finds the clusters, SAM 2.1 outlines and follows each one. A cluster keeps one id for as long as it is in view. |
| YOLO26, YOLO11 | Tracker ids (ByteTrack). An id counts once it has been followed for a second. |
| SAM 3, SAM 3.1 | The model's own ids, from the text prompt "grape cluster". |

The totals land in `data/results/`:

- `cluster_counts.csv`: one line per video and model.
- `cluster_counts_by_video.csv`: one row per video, one column per model.
- `sam2/`, `yolo26/`, `yolo11/`, `sam3/`, `sam3_1/`: each pipeline's overlay video and per-frame counts.

A video that has its total is skipped, so run the script again when new video arrives or a job ran out of time. A total is the number of clusters a model followed, not a hand count: count one row by hand and compare before relying on the numbers.

### Watch the jobs

```bash
squeue --me
```

`PD` is waiting, `R` is running, and a job that is no longer listed has finished. Jobs keep running when the terminal is closed or the connection drops. A job that asks for 4 hours or less usually starts within minutes. Stop one with `scancel` and its number from the first column.

The end of the newest log:

```bash
tail -n 20 "$(ls -t ~/Capstone-MSU_Fruit/*_pipeline/*.out | head -1)"
```

Each pipeline's logs are in its own folder, named `<job name>-<job id>.out`: `fruit-…` in `sam2_pipeline`, `sam3-…`, `sam3_1-…`, `yolo26-…`, `yolo11-…` in theirs.

### One pipeline at a time

Each pipeline's README has its own steps as blocks to paste: [SAM 2.1](#sam-21-pipeline), [SAM 3](#sam-3-pipeline), [SAM 3.1](#sam-31-pipeline), [YOLO26](#yolo26-pipeline), [YOLO11](#yolo11-pipeline). They use two kinds of command:

- **In the terminal**, for what takes a few minutes: `source hpcc/activate.sh` in the pipeline's folder, then the command. A process on a development node gets two hours of CPU time; the activate scripts keep a program to 8 CPU threads so that this lasts.
- **As a batch job**, for training and for whole folders of video: `sbatch hpcc/run.sb …` from the pipeline's folder.

### Getting a better model

- **Labels decide everything.** More labelled clips from more videos (rows, light, cameras) beat more training steps on the same clips.
- **SAM 2.1:** start at 3000 steps; then `--lr`, `--encoder yes` (trains the image backbone too; needs a 40 GB GPU) and `--init` (continue from an earlier run). `python fruit.py train --help`.
- **YOLO on our clips:** `train-detector` trains at 1280 px for 100 epochs; next try `--epochs 200`, or a bigger base (`--base yolo26s-seg.pt`).
- **SAM 3 and SAM 3.1:** `finetune.py --epochs 40`, or `--resolution 768` if the GPU runs out of memory.
- **The video pipeline with both fine-tuned:** [`sam2_pipeline/README.md`](#sam-21-pipeline), step 5. To score that pair on the leaderboard, switch on the commented entry at the end of `sam2_pipeline/configs/lineup.yaml`.

### If something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded: "Miniforge3"` | You are on a gateway node. `ssh dev-amd24-h200`, then paste the block again. |
| `-bash: dev-amd24-h200: command not found` | It needs `ssh` in front: `ssh dev-amd24-h200`. |
| `Unable to open file hpcc/run.sb`, or `can't open file …/train.py` | The wrong folder. Every block starts with the `cd` line it needs; paste the whole block. |
| `Killed` in the middle of a run in the terminal | The process used up its two hours of CPU time. Start from `source hpcc/activate.sh` in the pipeline's folder, which limits the CPU threads, or submit the run as a batch job. |
| A job stays `PD` for a long time | Jobs that ask for more than 4 hours wait much longer. The scripts here ask for 4 hours or less; a job that runs out of time continues when it is submitted again. |
| `conda` stops with `No module named …` before anything of ours runs | Check the cluster's own conda: `(module purge; module load Miniforge3; conda --version)`. If that fails too, it is the HPCC, not your setup: try later, or write to ICER through <https://contact.icer.msu.edu>. |
| `401` or `GatedRepoError` from `hf download` | Access to that model is not approved yet, or `hf auth login` was not done (step 1). |

---

## Labelling guide

*From `LABELLING.md`.*

You draw one box around each grape cluster, once. SAM 2.1 turns the box into an outline and follows the cluster through the clip. You check every frame and fix what is wrong. Labelled clips are what every model is fine-tuned on and scored against.

### The rule

A cluster is what the vine carries when the video was shot: early in the season a flower cluster, a small bumpy green bunch on a young shoot; later a bunch of grapes.

1. One cluster, one box. Two clusters that touch are two if you can tell them apart. A cluster cut in two by a leaf, a shoot or a wire is one.
2. Label every cluster on the row in front of the camera, also the small and the half-hidden ones. Leave out the rows behind it.
3. A cluster keeps its outline and its number through the whole clip. If an outline jumps to another cluster, fix it.

### Open the labelling screen

On the MSU HPCC, after `bash hpcc/setup.sh` ([hpcc/README.md](#hpcc-instructions)) and with video in `data/videos/`:

1. <https://ondemand.hpcc.msu.edu> ▸ *Interactive Apps* ▸ *Jupyter*. Jupyter location: *Conda Environment using Miniforge3 module*, environment `~/envs/fruit_sam2`. *Advanced Options*: 1 GPU. 2 to 4 hours. *Launch*, then *Connect*.
2. Open `Capstone-MSU_Fruit/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb`.
3. In cell **A · Settings** untick `DEMO` and set `PROJECT_DIR` to `~/Capstone-MSU_Fruit/data`.
4. Click the cell **Step 3b · Label**, choose *Run ▸ Run All Above Selected Cell*, then run the cell itself. The cells above it cut the videos into clips (16 frames each) the first time.

On a laptop or in Colab the same notebook works: [sam2_pipeline/INSTALL.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/INSTALL.md).

### Label a clip

| To | Do this |
|---|---|
| Pick a clip | The **clip** menu: ⬜ to do, ✏️ draft, ✅ done. A name at the end means a teammate is on that clip: take another. |
| Label a cluster | Drag a box around it, on any frame. |
| Move through the clip | **◀ ▶** or the frame slider. A cluster that comes into view later: box it on that frame. |
| Fix a wrong outline | *a click will:* **Select**, click the cluster, press **Re-box selected**, drag the right box on that frame. |
| Nudge an outline | *a click will:* **Add to selected** or **Remove from selected**, then click the spot. |
| Remove a cluster | Select it, press **Delete selected**. |
| Take back the last change | **Undo**. |
| Finish the clip | Every cluster right in every frame: press **Clip checked: mark done**. |

Every change is saved at once. Only clips marked done are used.

### How many

The `val` clips first (they are at the top of the menu): at least 8, from at least 2 videos. Every model is scored on them. Then as many `train` clips as you can.

How many clips are done, in a terminal on a development node ([hpcc/README.md](#hpcc-instructions), "Log in"):

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py status
```

### A head start

When a model already finds the clusters in this footage ([sam2_pipeline/README.md](#sam-21-pipeline), step 6, shows its boxes on them), let it draft the labels. In **Step 3a** tick `DRAFT_LABELS` and run the cell: the model boxes the clusters in every clip that has no labels yet. Then open each clip in Step 3b, add the clusters it missed, fix the wrong ones and mark it done.

Footage no model has learned, such as early-season video, has no head start at first. Label the val clips and a first batch of train clips by hand and fine-tune (next section). Then set `DRAFT_WEIGHTS` in Step 3a to the fine-tuned detector, `runs/yolo26_grapes_v1/weights/best.pt`, and let it draft the rest.

### Then

Fine-tune every model on the labelled clips and score them all ([hpcc/README.md](#hpcc-instructions), step 4):

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1
```

---

## SAM 2.1 pipeline

*From `sam2_pipeline/README.md`. "This folder" is `sam2_pipeline/`; paths such as `model/` and `../data/` start there.*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb)

Label grape clusters in our video, fine-tune SAM 2.1 on the labels, score every model on the same held-out clips with the same rules, and count the clusters in a video.

```
videos ──▶ clips ──▶ labels ──▶ fine-tune ──▶ leaderboard ──▶ count the clusters in new video
```

On the MSU HPCC: the steps below. In Google Colab or on a laptop: [INSTALL.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/INSTALL.md), with a 10-minute demo.

### On the MSU HPCC

Set up once: [`../hpcc/README.md`](#hpcc-instructions), step 1. Every block below can be pasted as it is into a terminal on a development node: log in, then `ssh dev-amd24-h200`.

#### 1. See what is in the project

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py status
```

It lists the videos, the clips cut from them, how many are labelled, the fine-tuned models and the last leaderboard.

#### 2. Cut new video into clips, then label

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py clips
```

It cuts 16-frame clips from every video under `data/videos/` that has none yet. Then label them: [`../LABELLING.md`](#labelling-guide).

#### 3. Fine-tune on the labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v1 --steps 3000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --name yolo26_grapes_v1
```

Two batch jobs: SAM 2.1 from Meta's pretrained weights (→ `data/models/sam2.1_base_plus_grapes_v1.pt`) and the YOLO26 detector from its WGISD model (→ `data/runs/yolo26_grapes_v1/weights/best.pt`). `squeue --me` lists them until they are done; the logs are `fruit-<job id>.out` in this folder. A run name is trained once: after labelling more clips, use a new name such as `grapes_v2`.

#### 4. The leaderboard

When `squeue --me` no longer lists the training jobs:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

A few minutes, in the terminal. It scores every model on the labelled val clips, prints two tables and saves them as `data/results/leaderboard_val.csv`:

- **Prompted** (SAM 2.1): shown each cluster once by a box or a click, must follow it. Headline **J&F** (0–100).
- **Automatic** (YOLO26, YOLO11, YOLOE, detector → SAM 2.1, SAM 3, SAM 3.1): gets only pixels. Headline **F1** at IoU 0.5, plus precision, recall, count error, tracks found vs true.

Row names say what each model is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<run>)`. SAM 3 and SAM 3.1 get their rows from step 2 of their own READMEs.

#### 5. Count the clusters in a video

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

#### 6. A model finds nothing in a video

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
| Footage unlike the photos the detector learned from, such as early-season video, where the clusters are flower clusters | No size has boxes, or a few low-scoring boxes sit on leaves | Label clips of this footage ([`../LABELLING.md`](#labelling-guide)) and fine-tune |

#### More about the jobs

`run.sb` asks for one A100 for 4 hours; override per job, for example `sbatch --gpus=v100:1 --time=02:00:00 hpcc/run.sb evaluate`. A training run that hits the time limit resumes when the same line is submitted again. Everything after `run.sb` goes to `fruit.py` (`python fruit.py --help` lists the commands); `bash --login hpcc/run.sb …` runs the same thing in the terminal instead of the queue. The project folder and the SAM 2.1 size are set in `hpcc/config.sh`.

### Reference

#### Reading the results

- Fine-tuning is not guaranteed to help. Compare fine-tuned and pretrained on the same val clips, and watch the comparison video before trusting a gap of a point or two.
- Labels drawn with SAM 2.1's help flatter its *box* rows; lean on the click rows and the automatic table.
- A cluster total is the number of clusters a model followed. A cluster that is hidden and comes back is counted twice, a missed one not at all. The leaderboard's *tracks found / true* column is that check on the labelled clips; count one row by hand before relying on a video's total.
- A cluster count is not yield; converting counts to kilograms needs a calibration against hand-harvested vines.

#### Scoring any model

The leaderboard scores every model in `configs/lineup.yaml` plus every model it finds in the project: SAM 2.1 weights in `model/` and `data/models/`, detectors from `train-detector` in `data/runs/`, SAM 3 and SAM 3.1 masks in `data/predictions/`. The table keeps the rows of earlier runs on the same clips, so models can be scored one at a time; `evaluate --fresh` starts it over.

To add a model, add an entry under `models:` in `configs/lineup.yaml`. Paths are relative to the project folder, then to the repository; an entry whose file is missing is skipped.

```yaml
  - {name: my YOLO-seg, type: ultralytics, weights: runs/yolo_detector/weights/best.pt}
  - {name: YOLO26 → SAM 2.1 fine-tuned, type: detect_then_sam2, checkpoint: sam2_pipeline/model/sam2.1_base_plus_grapes_v1.pt,
     detector: {type: ultralytics, weights: yolo26_pipeline/model/best.pt, conf: 0.25}}
  - {name: masks from another program, type: folder, path: predictions/my_masks, tracks: true}
  - {name: my model, type: python, class: ../examples/my_model.py:MyModel}
```

#### Labels made elsewhere

Roboflow, CVAT, Label Studio or YOLO exports, with the path of your export in place of `export/…`:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py import-coco export/train/_annotations.coco.json
python fruit.py import-yolo export/train/images export/train/labels
python fruit.py approve --all-drafts
```

COCO or YOLO outlines are used as drawn. Box-only labels get a SAM 2.1 mask and stay drafts until checked in the labelling screen or approved with the third line. The split is by source video, read off the file names.

#### Files

| File | What it is |
|---|---|
| `fruit.py` | The pipeline as a command line: `status`, `clips`, `train`, `train-detector`, `evaluate`, `video`, `counts`, `check`, … (`python fruit.py --help`). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a job, `setup_env.sh` builds the environment, `config.sh` holds the paths. |
| `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` | The whole pipeline as a notebook, with the labelling screen. Colab, or Jupyter on the HPCC. |
| `configs/lineup.yaml` | The models the leaderboard scores. Add any model here. |
| `model/` | The fine-tuned SAM 2.1 the team ships ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/model/README.md)). |
| `INSTALL.md`, `environment.yml` | Install guide and conda environment for a laptop. |
| `demo.ipynb` | 10-minute demo on public grapes: score pretrained SAM 2.1, fine-tune it, score again. |
| `count_test.ipynb` | Our YOLO26 → SAM 2.1 counts against hand counts on 27 WGISD photos. |
| `tests/test_install.py` | One-minute install check. |
| `fruit_sam2.py` | The notebook's tool cells as one module. Generated; do not edit. |
| `source/` | The notebooks' cells and `build.py`. Edit a cell, run `python source/build.py`, commit the regenerated files. |
| `examples/my_model.py` | Template for scoring your own model. |

**The project folder** is [`../data/`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/data/README.md) (every `fruit.py` command uses it unless you pass `--project`): `videos/`, `dataset/` (clips and labels), `predictions/`, `models/`, `results/`, `runs/`. In Colab it is on Google Drive; the demo keeps to `data/demo/`.

#### Words used here

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

### When something goes wrong

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

---

## SAM 3 pipeline

*From `sam3_pipeline/README.md`. "This folder" is `sam3_pipeline/`; paths such as `model/` and `../data/` start there.*

SAM 3 (Meta, November 2025) segments and tracks every object that matches a text prompt ("grape cluster"), with no boxes drawn by anyone. SAM 3.1 ([`../sam3_1_pipeline/`](#sam-31-pipeline)) is the same model family with a faster multi-object tracker; this folder keeps the original so the two can be compared on the leaderboard. This folder counts the clusters in a video with it, scores it on our labelled clips next to every other model, and fine-tunes it on those clips. **MUST HAVE HUGGING FACE ACCESS TO THE GATED REPO TO RUN**

```
../data/videos/<file>            ──▶ run.py video ──▶ ../data/results/sam3/<file>_sam3.mp4 + .csv, the total in ../data/results/cluster_counts.csv
../data/dataset (labelled clips) ──▶ run.py clips ──▶ ../data/predictions/sam3_text/ ──▶ the leaderboard in ../sam2_pipeline
../data/dataset (labelled clips) ──▶ finetune.py ──▶ model/sam3_<run>.pt ──▶ run.py clips ──▶ ../data/predictions/sam3_<run>/ ──▶ the leaderboard
```

### On the MSU HPCC

Set up once: [`../hpcc/README.md`](#hpcc-instructions), step 1, which builds the environment and puts Meta's weights in `model/`. Every block below can be pasted as it is into a terminal on the development node: log in, then `ssh dev-amd24-h200`.

#### 1. Count the clusters in a video

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
python run.py video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

A few minutes for a 20-second video. It ends with `Clusters counted: …`, writes `../data/results/sam3/<video>_sam3.mp4` (the video with every cluster tinted) and `.csv` (`frame`, `visible_now`, `clusters_so_far`), and adds the total to `../data/results/cluster_counts.csv`, the file every pipeline adds to. A total is the number of clusters the model followed, not a hand count, and not a yield estimate.

Every video of a folder, as a batch job (videos that have a total are skipped):

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
sbatch --time=04:00:00 hpcc/run.sb run.py video ../data/videos/2026/Grape--DJIOsmo
```

#### 2. Score the released model on our labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
python run.py clips --checkpoint release
```

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

The first block writes the model's masks for the labelled val clips to `../data/predictions/sam3_text/`. The second is the leaderboard of every model, a few minutes in the terminal: this model is the row `SAM 3 · text 'grape cluster' · zero-shot`.

#### 3. Fine-tune on our labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
sbatch --time=04:00:00 hpcc/run.sb finetune.py --run grapes_v1
```

One batch job, about 40 minutes for a few labelled clips. To see how far it is (the first number in brackets is the epoch, 0 to 19):

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
squeue --me
grep "Train Epoch" "$(ls -t sam3-*.out | head -1)" | tail -n 1
```

When `squeue --me` no longer lists the job, its log ends with `merged … detector tensors` and the model is `model/sam3_grapes_v1.pt`. Score it:

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
python run.py clips
```

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

Its row is `SAM 3 · text 'grape cluster' → fine-tuned on our clips (grapes_v1)`. From now on `run.py` uses the fine-tuned model, the newest file in `model/`, so step 1 counts with it; `--checkpoint release` after the command goes back to Meta's weights. A run name is trained once: after labelling more clips, use a new name such as `grapes_v2`.

### Reference

#### `run.py`

- **Weights**: the newest `.pt` in `model/` (Meta's release after the setup, a fine-tuned model after `finetune.py`). `--checkpoint release` forces Meta's weights, `--checkpoint <file>` a particular file.
- **Options**, before or after the command name: `--text "green grapes"` (one phrase per prompt), `--prob 0.4` (output threshold; default 0.5), `--data <folder>` (project folder; default `../data` or `$PROJECT_DIR`).
- **`video`**: the video goes through the model in stretches of 10 seconds that share one frame; a cluster keeps its id from one stretch to the next when its masks on the shared frame overlap. The total is the number of ids seen in at least two frames. `--start 60 --seconds 30` runs a stretch for a quick look; only a whole video adds its total. In a folder, `--overwrite` counts again the videos that have a total. A video's overlay and `.csv` are replaced by a later run with another model or prompt; each model's total stays in `cluster_counts.csv`.
- **`clips`**: each run writes `../data/predictions/<name>/` with a `recipe.json`, and the leaderboard adds a row for it by itself.

#### `finetune.py`

It exports the labelled clips as a COCO dataset (`../data/sam3_dataset/`: train clips → `train/`, val clips → `test/`), runs Meta's trainer on it starting from the released weights (log, curves and checkpoint in `../data/runs/sam3_grapes_v1/`), and merges the fine-tuned detector into the released video checkpoint: `model/sam3_grapes_v1.pt` with a `.json` recipe. The trainer fine-tunes the detector, the part that finds clusters from the text; the tracker that follows them through the video stays as released. A job that runs out of time continues from its last finished epoch when the same line is submitted again.

| Argument | Default | Meaning |
|---|---|---|
| `--run` | required | Name of the run |
| `--epochs` | `20` | Training epochs; an epoch is one pass over the training images |
| `--epoch-size` | `100` | At least this many images per epoch; with fewer training images, each is shown several times |
| `--resolution` | `1008` | Input size; lower it if the GPU runs out of memory |
| `--batch` | `1` | Batch size |
| `--every` | `2` | Use every n-th frame of a clip |
| `--text` | `grape cluster` | The category name, which is the text prompt the model learns |
| `--dry-run` | | Export the dataset and write the recipe without training |
| `--merge-only <checkpoint>` | | Merge an already trained `checkpoints/checkpoint.pt` into the released weights |

With the defaults that is 2,000 training steps or more, about a second each on an A100 or H200. The fine-tuned file is 3 GB and ignored by git; share it through a GitHub Release ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam3_pipeline/model/README.md)).

#### Files

| File | What it does |
|---|---|
| `run.py` | `video`: counts the clusters in a video, or in every video of a folder. `clips`: masks for the labelled val clips. |
| `finetune.py` | Fine-tunes the model on our labelled clips with Meta's trainer and writes the result to `model/`. |
| `finetune.yaml` | Meta's fine-tuning recipe with our dataset and settings (filled in by `finetune.py`). |
| `meta_trainer.py` | Starts Meta's trainer for `finetune.py`, with the one correction it needs to train. |
| `pipeline.py` | The names every script uses: model version, weights, data folder, where predictions and results go. |
| `model/` | The weights: Meta's release and our fine-tuned checkpoints ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam3_pipeline/model/README.md)). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a batch job, `setup_env.sh` builds the environment. |
| `requirements.txt` | Python packages (Meta's `sam3` package with its trainer). |

### On another computer

It needs an NVIDIA GPU with CUDA 12.6 or newer (not a CPU, not a Mac), Python 3.12 or newer, and access to the weights: request it at [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face and wait for the approval email. On a Linux machine, not on the HPCC:

```bash
cd Capstone-MSU_Fruit/sam3_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
hf auth login
```

```bash
hf download facebook/sam3 sam3.pt --local-dir model
```

Use the CUDA build of PyTorch that matches the machine (`cu126`, or `cu128` on newer cards). Then the commands are the ones above without the `source hpcc/activate.sh` and `sbatch` parts: `python run.py video <video>`, `python run.py clips`, `python finetune.py --run grapes_v1`, and `python fruit.py evaluate` in `../sam2_pipeline`.

Colab: *Runtime ▸ Change runtime type ▸ T4 GPU*, then `!git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit/sam3_pipeline && pip install -q -r requirements.txt`, then `from huggingface_hub import login; login()`. Copy `data/dataset/` from Drive into the clone's `data/`.

### If something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded` | You are on a gateway node: `ssh dev-amd24-h200`, then paste the block again. |
| `can't open file …/run.py`, or `No module named 'sam3'` | Paste the whole block: it starts with the `cd` and `source hpcc/activate.sh` lines this folder needs. |
| `401` / `GatedRepoError` on download | Access to `facebook/sam3` not approved yet, or `hf auth login` not done on this machine. |
| `Killed` in the middle of a run in the terminal | The process used up the two hours of CPU time it gets on a development node. Submit the run as a batch job: `sbatch --time=04:00:00 hpcc/run.sb` and then the command without `python`. |
| `CUDA out of memory` | `run.py`: shorten `--seconds`. `finetune.py`: `--resolution 768`, or a bigger GPU. |
| `No module named sam3.train`, `psutil`, `einops`, `pycocotools` or `pkg_resources` | `python -m pip install -r requirements.txt`: it holds the `[train]` extra and the packages `sam3` imports without declaring them. |
| `No detector weights matched` after training | The trained checkpoint is kept in `../data/runs/sam3_<run>/checkpoints/`; open an issue with the first lines of `unmatched_tensors` from the `.json`. |
| Masks look right but the leaderboard shows 0 tracks | Check that `../data/dataset/clips.json` lists the clip as `val` and `done`. |

Licence: Meta's SAM License. The clips and videos belong to National Grape.

---

## SAM 3.1 pipeline

*From `sam3_1_pipeline/README.md`. "This folder" is `sam3_1_pipeline/`; paths such as `model/` and `../data/` start there.*

SAM 3.1 (Meta, March 2026) segments and tracks every object that matches a text prompt ("grape cluster"), with no boxes drawn by anyone. It is SAM 3 with Object Multiplex, a faster tracker for many objects at once; the original SAM 3 is [`../sam3_pipeline/`](#sam-3-pipeline). This folder counts the clusters in a video with it, scores it on our labelled clips next to every other model, and fine-tunes it on those clips. **MUST HAVE HUGGING FACE ACCESS TO THE GATED REPO TO RUN**

```
../data/videos/<file>            ──▶ run.py video ──▶ ../data/results/sam3_1/<file>_sam3_1.mp4 + .csv, the total in ../data/results/cluster_counts.csv
../data/dataset (labelled clips) ──▶ run.py clips ──▶ ../data/predictions/sam3_1_text/ ──▶ the leaderboard in ../sam2_pipeline
../data/dataset (labelled clips) ──▶ finetune.py ──▶ model/sam3_1_<run>.pt ──▶ run.py clips ──▶ ../data/predictions/sam3_1_<run>/ ──▶ the leaderboard
```

### On the MSU HPCC

Set up once: [`../hpcc/README.md`](#hpcc-instructions), step 1, which builds the environment and puts Meta's weights in `model/`. Every block below can be pasted as it is into a terminal on the development node: log in, then `ssh dev-amd24-h200`.

#### 1. Count the clusters in a video

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
source hpcc/activate.sh
python run.py video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

A few minutes for a 20-second video. It ends with `Clusters counted: …`, writes `../data/results/sam3_1/<video>_sam3_1.mp4` (the video with every cluster tinted) and `.csv` (`frame`, `visible_now`, `clusters_so_far`), and adds the total to `../data/results/cluster_counts.csv`, the file every pipeline adds to. A total is the number of clusters the model followed, not a hand count, and not a yield estimate.

Every video of a folder, as a batch job (videos that have a total are skipped):

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
sbatch --time=04:00:00 hpcc/run.sb run.py video ../data/videos/2026/Grape--DJIOsmo
```

#### 2. Score the released model on our labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
source hpcc/activate.sh
python run.py clips --checkpoint release
```

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

The first block writes the model's masks for the labelled val clips to `../data/predictions/sam3_1_text/`. The second is the leaderboard of every model, a few minutes in the terminal: this model is the row `SAM 3.1 · text 'grape cluster' · zero-shot`.

#### 3. Fine-tune on our labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
sbatch --time=04:00:00 hpcc/run.sb finetune.py --run grapes_v1
```

One batch job, about 40 minutes for a few labelled clips. To see how far it is (the first number in brackets is the epoch, 0 to 19):

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
squeue --me
grep "Train Epoch" "$(ls -t sam3_1-*.out | head -1)" | tail -n 1
```

When `squeue --me` no longer lists the job, its log ends with `merged … detector tensors` and the model is `model/sam3_1_grapes_v1.pt`. Score it:

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
source hpcc/activate.sh
python run.py clips
```

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

Its row is `SAM 3.1 · text 'grape cluster' → fine-tuned on our clips (grapes_v1)`. From now on `run.py` uses the fine-tuned model, the newest file in `model/`, so step 1 counts with it; `--checkpoint release` after the command goes back to Meta's weights. A run name is trained once: after labelling more clips, use a new name such as `grapes_v2`.

### Reference

#### `run.py`

- **Weights**: the newest `.pt` in `model/` (Meta's release after the setup, a fine-tuned model after `finetune.py`). `--checkpoint release` forces Meta's weights, `--checkpoint <file>` a particular file.
- **Options**, before or after the command name: `--text "green grapes"` (one phrase per prompt), `--prob 0.4` (output threshold; default 0.5), `--max-objects 128` (how many objects it can follow at once; default 64), `--data <folder>` (project folder; default `../data` or `$PROJECT_DIR`).
- **`video`**: the video goes through the model in stretches of 10 seconds that share one frame; a cluster keeps its id from one stretch to the next when its masks on the shared frame overlap. The total is the number of ids seen in at least two frames. `--start 60 --seconds 30` runs a stretch for a quick look; only a whole video adds its total. In a folder, `--overwrite` counts again the videos that have a total. A video's overlay and `.csv` are replaced by a later run with another model or prompt; each model's total stays in `cluster_counts.csv`.
- **`clips`**: each run writes `../data/predictions/<name>/` with a `recipe.json`, and the leaderboard adds a row for it by itself.

#### `finetune.py`

It exports the labelled clips as a COCO dataset (`../data/sam3_dataset/`: train clips → `train/`, val clips → `test/`), runs Meta's trainer on it starting from the released weights (log, curves and checkpoint in `../data/runs/sam3_1_grapes_v1/`), and merges the fine-tuned detector into the released video checkpoint: `model/sam3_1_grapes_v1.pt` with a `.json` recipe. The trainer fine-tunes the detector, the part that finds clusters from the text; the tracker that follows them through the video stays as released. A job that runs out of time continues from its last finished epoch when the same line is submitted again.

| Argument | Default | Meaning |
|---|---|---|
| `--run` | required | Name of the run |
| `--epochs` | `20` | Training epochs; an epoch is one pass over the training images |
| `--epoch-size` | `100` | At least this many images per epoch; with fewer training images, each is shown several times |
| `--resolution` | `1008` | Input size; lower it if the GPU runs out of memory |
| `--batch` | `1` | Batch size |
| `--every` | `2` | Use every n-th frame of a clip |
| `--text` | `grape cluster` | The category name, which is the text prompt the model learns |
| `--dry-run` | | Export the dataset and write the recipe without training |
| `--merge-only <checkpoint>` | | Merge an already trained `checkpoints/checkpoint.pt` into the released weights |

With the defaults that is 2,000 training steps or more, about a second each on an A100 or H200. The fine-tuned file is 3 GB and ignored by git; share it through a GitHub Release ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam3_1_pipeline/model/README.md)).

#### Files

| File | What it does |
|---|---|
| `run.py` | `video`: counts the clusters in a video, or in every video of a folder. `clips`: masks for the labelled val clips. |
| `finetune.py` | Fine-tunes the model on our labelled clips with Meta's trainer and writes the result to `model/`. |
| `finetune.yaml` | Meta's fine-tuning recipe with our dataset and settings (filled in by `finetune.py`). |
| `meta_trainer.py` | Starts Meta's trainer for `finetune.py`, with the one correction it needs to train. |
| `pipeline.py` | The names every script uses: model version, weights, data folder, where predictions and results go. |
| `model/` | The weights: Meta's release and our fine-tuned checkpoints ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam3_1_pipeline/model/README.md)). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a batch job, `setup_env.sh` builds the environment. |
| `requirements.txt` | Python packages (Meta's `sam3` package with its trainer). |

### On another computer

It needs an NVIDIA GPU with CUDA 12.6 or newer (not a CPU, not a Mac), Python 3.12 or newer, and access to the weights: request it at [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) on Hugging Face and wait for the approval email. On a Linux machine, not on the HPCC:

```bash
cd Capstone-MSU_Fruit/sam3_1_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
hf auth login
```

```bash
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir model
```

Use the CUDA build of PyTorch that matches the machine (`cu126`, or `cu128` on newer cards). Then the commands are the ones above without the `source hpcc/activate.sh` and `sbatch` parts: `python run.py video <video>`, `python run.py clips`, `python finetune.py --run grapes_v1`, and `python fruit.py evaluate` in `../sam2_pipeline`.

Colab: *Runtime ▸ Change runtime type ▸ T4 GPU*, then `!git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit/sam3_1_pipeline && pip install -q -r requirements.txt`, then `from huggingface_hub import login; login()`. Copy `data/dataset/` from Drive into the clone's `data/`.

### If something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded` | You are on a gateway node: `ssh dev-amd24-h200`, then paste the block again. |
| `can't open file …/run.py`, or `No module named 'sam3'` | Paste the whole block: it starts with the `cd` and `source hpcc/activate.sh` lines this folder needs. |
| `401` / `GatedRepoError` on download | Access to `facebook/sam3.1` not approved yet, or `hf auth login` not done on this machine. |
| `Killed` in the middle of a run in the terminal | The process used up the two hours of CPU time it gets on a development node. Submit the run as a batch job: `sbatch --time=04:00:00 hpcc/run.sb` and then the command without `python`. |
| `CUDA out of memory` | `run.py`: lower `--max-objects`, shorten `--seconds`. `finetune.py`: `--resolution 768`, or a bigger GPU. |
| `No module named sam3.train`, `psutil`, `einops`, `pycocotools` or `pkg_resources` | `python -m pip install -r requirements.txt`: it holds the `[train]` extra and the packages `sam3` imports without declaring them. |
| `No detector weights matched` after training | The trained checkpoint is kept in `../data/runs/sam3_1_<run>/checkpoints/`; open an issue with the first lines of `unmatched_tensors` from the `.json`. |
| Masks look right but the leaderboard shows 0 tracks | Check that `../data/dataset/clips.json` lists the clip as `val` and `done`. |

Licence: Meta's SAM License. The clips and videos belong to National Grape.

---

## YOLO26 pipeline

*From `yolo26_pipeline/README.md`. "This folder" is `yolo26_pipeline/`; paths such as `model/` and `../data/` start there.*

YOLO26n-seg finds and outlines grape clusters in a photo or a video frame. This folder trains it, scores it and counts the clusters in a video with it. It reads everything from `../data/` and writes training runs to `../data/runs/yolo26/` and everything else to `../data/results/yolo26/`. The model it ships is `model/best.pt`.

```
../data/wgisd_yolo ──▶ train.py ──▶ ../data/runs/yolo26/<run>/weights/best.pt ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

On a laptop: [INSTALL.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/yolo26_pipeline/INSTALL.md).

### On the MSU HPCC

Set up once: [`../hpcc/README.md`](#hpcc-instructions), step 1. The SAM 2.1 environment it builds runs this folder too. Every block below can be pasted as it is into a terminal on a development node: log in, then `ssh dev-amd24-h200`.

#### 1. Score the shipped model

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
source hpcc/activate.sh
python evaluate.py
python count_test.py
```

About a minute each. `evaluate.py` prints box and mask mAP on the 27 held-out WGISD test photos. `count_test.py` compares cluster counts with hand counts on the same photos and writes `../data/results/yolo26/count_test.csv` and the annotated photos in `../data/results/yolo26/count_test/`. These are scores on WGISD photos; the leaderboard of the SAM 2.1 pipeline scores this model on our own labelled clips.

#### 2. Count the clusters in a video

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

#### 3. Train on WGISD

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

To ship a run, copy its `weights/best.pt` over `model/best.pt` and update [model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/yolo26_pipeline/model/README.md).

#### 4. Fine-tune on our own clips

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

### Reference

#### `train.py`

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | `../data/wgisd_yolo/grape.yaml` | Dataset ([`../data/README.md`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/data/README.md)) |
| `--model` | `yolo26n-seg.pt` | Starting weights: an Ultralytics name (downloaded) or a file such as `model/best.pt` |
| `--epochs` | `50` | Epochs (stops early after 20 without improvement) |
| `--imgsz` | `512` | Image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--name` | `wgisd` | Run folder `../data/runs/yolo26/<name>/`; a number is added when it exists |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

#### `evaluate.py` and `count_test.py`

`evaluate.py`: `--model <weights>` scores another checkpoint, `--split val` the validation split; several `--model` weights write a side-by-side table to `../data/results/yolo26/evaluate_<split>.csv`; the plots are in `../data/runs/yolo26/eval_test_<name>/`. `count_test.py`: `--conf` (default 0.25) trades missed clusters for false ones. Both score a model at the image size it was trained at.

#### `video.py`

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

#### Files

| File | What it does |
|---|---|
| `train.py` | Trains the model: on the public WGISD photos by default, on our own labels with `--data`. |
| `evaluate.py` | Box and mask mAP on the WGISD test photos; several weights make a table. |
| `count_test.py` | Cluster counts against hand counts on the 27 WGISD test photos. |
| `video.py` | Counts the clusters in a video, or in every video of a folder: annotated video, per-frame counts, the total. |
| `pipeline.py` | The names every script uses (the model file, the data folder, where runs and results go) and `load()`, which opens a weights file and says what it is. |
| `model/` | The model this pipeline ships ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/yolo26_pipeline/model/README.md)). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a batch job, `setup_env.sh` for an environment of its own. |
| `tests/` | `python -m unittest discover -s tests` |
| `requirements.txt` | Python packages. |

The model detects grape clusters, not berries. Outside the HPCC, fine-tuning on our own labels is two commands: `python fruit.py export-yolo` in `../sam2_pipeline/` writes `../data/yolo_dataset/`, then `python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard` here. Labels made in Roboflow work the same way: export them as a YOLO segmentation dataset and point `--data` at its `data.yaml`.

### If something goes wrong

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
| `No clusters found` | The table printed at the start shows what the model saw at every input size. Nothing at any size, or false positives and missed clusters: the model has only seen WGISD photos, so label our footage ([`../LABELLING.md`](#labelling-guide)) and fine-tune (step 4). |

Licences: [WGISD](https://github.com/thsant/wgisd) CC BY-NC 4.0, [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0. Repository: [`../LICENSE.txt`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/LICENSE.txt).

---

## YOLO11 pipeline

*From `yolo11_pipeline/README.md`. "This folder" is `yolo11_pipeline/`; paths such as `model/` and `../data/` start there.*

YOLO11n-seg finds and outlines grape clusters in a photo or a video frame; it is the model the team shipped in October. This folder trains it, scores it and counts the clusters in a video with it. It reads everything from `../data/` and writes training runs to `../data/runs/yolo11/` and everything else to `../data/results/yolo11/`. The model it ships is `model/best.pt`.

```
../data/wgisd_yolo ──▶ train.py ──▶ ../data/runs/yolo11/<run>/weights/best.pt ──▶ model/best.pt ──▶ evaluate.py · count_test.py · video.py
```

On a laptop: [INSTALL.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/yolo11_pipeline/INSTALL.md).

### On the MSU HPCC

Set up once: [`../hpcc/README.md`](#hpcc-instructions), step 1. The SAM 2.1 environment it builds runs this folder too. Every block below can be pasted as it is into a terminal on a development node: log in, then `ssh dev-amd24-h200`.

#### 1. Score the shipped model

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
source hpcc/activate.sh
python evaluate.py
python count_test.py
```

About a minute each. `evaluate.py` prints box and mask mAP on the 27 held-out WGISD test photos. `count_test.py` compares cluster counts with hand counts on the same photos and writes `../data/results/yolo11/count_test.csv` and the annotated photos in `../data/results/yolo11/count_test/`. These are scores on WGISD photos; the leaderboard of the SAM 2.1 pipeline scores this model on our own labelled clips.

#### 2. Count the clusters in a video

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
source hpcc/activate.sh
python video.py --video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

Every cluster gets an id that follows it from frame to frame (ByteTrack). The total is the number of ids followed for at least a second. It writes `../data/results/yolo11/<video>_track.mp4` (masks, boxes, ids, the count so far) and `<video>_track.csv`, and adds the total to `../data/results/cluster_counts.csv`, the file every pipeline adds to. A total is the number of clusters the tracker followed, not a hand count, and not a yield estimate.

Every video of a folder, as a batch job (videos that have a total are skipped):

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
sbatch hpcc/run.sb video.py --video-dir ../data/videos/2026/Grape--DJIOsmo
```

#### 3. Train on WGISD

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
source hpcc/activate.sh
python train.py
```

Ten minutes or less: 50 epochs on the public WGISD photos. It ends with `Best weights: …` and the path of the run folder under `../data/runs/yolo11/`, which also holds `results.csv`, the curves and the confusion matrix.

A stronger model (bigger network, larger images, more epochs), as a batch job:

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
sbatch hpcc/run.sb train.py --model yolo11s-seg.pt --imgsz 1024 --batch 8 --epochs 150 --name wgisd_s1024
```

When `squeue --me` no longer lists it, score that run:

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
source hpcc/activate.sh
python evaluate.py --model ../data/runs/yolo11/wgisd_s1024/weights/best.pt
```

To ship a run, copy its `weights/best.pt` over `model/best.pt` and update [model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/yolo11_pipeline/model/README.md).

#### 4. Fine-tune on our own clips

The labels live in the SAM 2.1 pipeline, so this is one of its jobs. It starts from the shipped model:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train-detector --base ../yolo11_pipeline/model/best.pt --name yolo11_grapes_v1
```

When `squeue --me` no longer lists it, the leaderboard scores it next to every other model:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

Count a video with it:

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
source hpcc/activate.sh
python video.py --video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4 --model ../data/runs/yolo11_grapes_v1/weights/best.pt
```

### Reference

#### `train.py`

| Argument | Default | Meaning |
| --- | --- | --- |
| `--data` | `../data/wgisd_yolo/grape.yaml` | Dataset ([`../data/README.md`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/data/README.md)) |
| `--model` | `yolo11n-seg.pt` | Starting weights: an Ultralytics name (downloaded) or a file such as `model/best.pt` |
| `--epochs` | `50` | Epochs (stops early after 20 without improvement) |
| `--imgsz` | `512` | Image size |
| `--batch` | `2` | Batch size; raise it on a GPU with more than 4 GB |
| `--name` | `wgisd` | Run folder `../data/runs/yolo11/<name>/`; a number is added when it exists |
| `--device` | `auto` | `auto`, `cpu`, or a CUDA device such as `0` |

#### `evaluate.py` and `count_test.py`

`evaluate.py`: `--model <weights>` scores another checkpoint, `--split val` the validation split; several `--model` weights write a side-by-side table to `../data/results/yolo11/evaluate_<split>.csv`; the plots are in `../data/runs/yolo11/eval_test_<name>/`. `count_test.py`: `--conf` (default 0.25) trades missed clusters for false ones. Both score a model at the image size it was trained at.

#### `video.py`

| Argument | Default | Meaning |
| --- | --- | --- |
| `--video` / `--video-dir` | one required | One video, or every video under a folder at any depth (subfolders kept in the output; unreadable and still-syncing files skipped; videos that have a total skipped unless `--overwrite`) |
| `--model` | `model/best.pt` | Weights |
| `--conf` | `0.30` | Minimum confidence |
| `--imgsz` | `auto` | The size a frame is shrunk to before the model sees it. `auto` tries every size on five frames of the video, prints what it finds at each and uses the one at which the model is most sure; a number fixes it |
| `--max-frames`, `--start-sec` | the whole video | With `--video`: a stretch of it, e.g. `--max-frames 300` for a quick look. Only a whole video adds its total to the table; a folder is always counted whole |
| `--mode` | `track` | `detect` draws what is in each frame, without ids or a total |
| `--output` | `../data/results/yolo11/<video>_<mode>.mp4` | Output video; the CSV gets the same name. A folder's videos go to `../data/results/yolo11/<folder>/` |

The input size matters because a model trained on photos only sees clusters of about the size they had in those photos: this one, about 40 to 100 px tall in the image it is given. Footage shot closer needs a smaller size, wider shots a larger one.

#### Files

| File | What it does |
|---|---|
| `train.py` | Trains the model: on the public WGISD photos by default, on our own labels with `--data`. |
| `evaluate.py` | Box and mask mAP on the WGISD test photos; several weights make a table. |
| `count_test.py` | Cluster counts against hand counts on the 27 WGISD test photos. |
| `video.py` | Counts the clusters in a video, or in every video of a folder: annotated video, per-frame counts, the total. |
| `pipeline.py` | The names every script uses (the model file, the data folder, where runs and results go) and `load()`, which opens a weights file and says what it is. |
| `model/` | The model this pipeline ships ([model/README.md](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/yolo11_pipeline/model/README.md)). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a batch job, `setup_env.sh` for an environment of its own. |
| `tests/` | `python -m unittest discover -s tests` |
| `requirements.txt` | Python packages. |

The model detects grape clusters, not berries. Outside the HPCC, fine-tuning on our own labels is two commands: `python fruit.py export-yolo` in `../sam2_pipeline/` writes `../data/yolo_dataset/`, then `python train.py --data ../data/yolo_dataset/data.yaml --model model/best.pt --imgsz 1280 --name vineyard` here. Labels made in Roboflow work the same way: export them as a YOLO segmentation dataset and point `--data` at its `data.yaml`.

### If something goes wrong

| Symptom | Fix |
| --- | --- |
| `You are currently using a gateway node`, or `module(s) could not be loaded` | You are on a gateway node: `ssh dev-amd24-h200`, then paste the block again. |
| `can't open file …/train.py`, or `No module named 'ultralytics'` | Paste the whole block: it starts with the `cd` and `source hpcc/activate.sh` lines this folder needs. |
| `Killed` in the middle of a run in the terminal | The process used up the two hours of CPU time it gets on a development node. Submit the run as a batch job: `sbatch hpcc/run.sb` and then the command without `python`. |
| A job log says a download failed | Fetch the base weights once on the development node: `source hpcc/activate.sh`, then `python -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt')"` in this folder, and submit again. |
| `Missing weights: …/model/best.pt` | Incomplete clone (`git status`), or train a model (step 3) and copy it there. |
| `Missing …/wgisd_yolo/grape.yaml` | `python ../data/wgisd_yolo/prepare_wgisd.py` |
| CUDA out of memory | Lower `--batch` (training) or `--imgsz` (video). |
| CUDA unavailable | The CPU is used. For a GPU, install a CUDA build of PyTorch (INSTALL.md). |
| `… is a base model` | A base model such as `yolo11n-seg.pt` has no grape class. Use `model/best.pt` or a trained run. |
| `No clusters found` | The table printed at the start shows what the model saw at every input size. Nothing at any size, or false positives and missed clusters: the model has only seen WGISD photos, so label our footage ([`../LABELLING.md`](#labelling-guide)) and fine-tune (step 4). |

Licences: [WGISD](https://github.com/thsant/wgisd) CC BY-NC 4.0, [Ultralytics](https://github.com/ultralytics/ultralytics) AGPL-3.0. Repository: [`../LICENSE.txt`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/LICENSE.txt).
