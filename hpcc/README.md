# hpcc/ — the workflow on the MSU HPCC

Five steps. Every block can be pasted into a terminal as it is.

## Log in

Use your NetID in place of `netid`. Paste the two lines one at a time: `ssh` swallows whatever is pasted after it.

```bash
ssh netid@hpcc.msu.edu
```

```bash
ssh dev-amd24-h200
```

The first line lands on a gateway node, which runs no software and refuses `sbatch`. The second goes to a development node, where everything below is typed: the prompt then shows `dev-amd24-h200`.

## 1. Set up, once per person

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

## 2. Bring in the video

[`data/README.md`](../data/README.md), "Video from SharePoint".

## 3. Label clips

[`LABELLING.md`](../LABELLING.md). Labels made in Roboflow or CVAT come in with `fruit.py import-coco` / `import-yolo` ([`sam2_pipeline/README.md`](../sam2_pipeline/README.md), "Labels made elsewhere").

## 4. Fine-tune every model and score them all

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

## 5. Count the clusters in every video

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

## Watch the jobs

```bash
squeue --me
```

`PD` is waiting, `R` is running, and a job that is no longer listed has finished. Jobs keep running when the terminal is closed or the connection drops. A job that asks for 4 hours or less usually starts within minutes. Stop one with `scancel` and its number from the first column.

The end of the newest log:

```bash
tail -n 20 "$(ls -t ~/Capstone-MSU_Fruit/*_pipeline/*.out | head -1)"
```

Each pipeline's logs are in its own folder, named `<job name>-<job id>.out`: `fruit-…` in `sam2_pipeline`, `sam3-…`, `sam3_1-…`, `yolo26-…`, `yolo11-…` in theirs.

## One pipeline at a time

Each pipeline's README has its own steps as blocks to paste: [SAM 2.1](../sam2_pipeline/README.md), [SAM 3](../sam3_pipeline/README.md), [SAM 3.1](../sam3_1_pipeline/README.md), [YOLO26](../yolo26_pipeline/README.md), [YOLO11](../yolo11_pipeline/README.md). They use two kinds of command:

- **In the terminal**, for what takes a few minutes: `source hpcc/activate.sh` in the pipeline's folder, then the command. A process on a development node gets two hours of CPU time; the activate scripts keep a program to 8 CPU threads so that this lasts.
- **As a batch job**, for training and for whole folders of video: `sbatch hpcc/run.sb …` from the pipeline's folder.

## Getting a better model

- **Labels decide everything.** More labelled clips from more videos (rows, light, cameras) beat more training steps on the same clips.
- **SAM 2.1:** start at 3000 steps; then `--lr`, `--encoder yes` (trains the image backbone too; needs a 40 GB GPU) and `--init` (continue from an earlier run). `python fruit.py train --help`.
- **YOLO on our clips:** `train-detector` trains at 1280 px for 100 epochs; next try `--epochs 200`, or a bigger base (`--base yolo26s-seg.pt`).
- **SAM 3 and SAM 3.1:** `finetune.py --epochs 40`, or `--resolution 768` if the GPU runs out of memory.
- **The video pipeline with both fine-tuned:** [`sam2_pipeline/README.md`](../sam2_pipeline/README.md), step 5. To score that pair on the leaderboard, switch on the commented entry at the end of `sam2_pipeline/configs/lineup.yaml`.

## If something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded: "Miniforge3"` | You are on a gateway node. `ssh dev-amd24-h200`, then paste the block again. |
| `-bash: dev-amd24-h200: command not found` | It needs `ssh` in front: `ssh dev-amd24-h200`. |
| `Unable to open file hpcc/run.sb`, or `can't open file …/train.py` | The wrong folder. Every block starts with the `cd` line it needs; paste the whole block. |
| `Killed` in the middle of a run in the terminal | The process used up its two hours of CPU time. Start from `source hpcc/activate.sh` in the pipeline's folder, which limits the CPU threads, or submit the run as a batch job. |
| A job stays `PD` for a long time | Jobs that ask for more than 4 hours wait much longer. The scripts here ask for 4 hours or less; a job that runs out of time continues when it is submitted again. |
| `conda` stops with `No module named …` before anything of ours runs | Check the cluster's own conda: `(module purge; module load Miniforge3; conda --version)`. If that fails too, it is the HPCC, not your setup: try later, or write to ICER through <https://contact.icer.msu.edu>. |
| `401` or `GatedRepoError` from `hf download` | Access to that model is not approved yet, or `hf auth login` was not done (step 1). |
