# hpcc/ — the workflow on the MSU HPCC

Five steps, all from `~/Capstone-MSU_Fruit`. GPU work goes to the queue as batch jobs: `squeue --me` shows them, `scancel <job id>` stops one.

## 1. Set up, once per person

On a development node (`ssh <netid>@hpcc.msu.edu`, then `ssh dev-amd20-v100`):

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit
bash hpcc/setup.sh
```

It does two things:

- **Shared data** (`hpcc/share_data.sh`): `data/videos` and `data/dataset` become links to `/mnt/research/MSU-Fruit/videos` and `/mnt/research/MSU-Fruit/dataset`, so the team has one copy of the video and one set of clips and labels. Videos or clips already in the clone move there first. Everything else in `data/` stays in the clone: each person's own runs and results.
- **Environments** (each pipeline's `hpcc/setup_env.sh`): SAM 2.1, which also runs the YOLO pipelines, then SAM 3 and SAM 3.1.

SAM 3 and SAM 3.1 then need Meta's weights: request access at [facebook/sam3](https://huggingface.co/facebook/sam3) and [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), and once approved run the two `hf` lines each setup printed.

## 2. Bring in the video

[`data/README.md`](../data/README.md), "Video from SharePoint".

## 3. Label clips

[`LABELLING.md`](../LABELLING.md). Labels made in Roboflow or CVAT come in with `fruit.py import-coco` / `import-yolo` ([`sam2_pipeline/README.md`](../sam2_pipeline/README.md), section 3).

## 4. Fine-tune every model and score them all

```bash
bash hpcc/run_all.sh grapes_v1 3000 100 20
```

The arguments are the run name, SAM 2.1 training steps, YOLO epochs and SAM 3 epochs; these are the defaults.

| Model | Starts from | Fine-tuned model |
|---|---|---|
| SAM 2.1 | Meta's pretrained `base_plus` | `data/models/sam2.1_base_plus_<name>.pt` |
| YOLO26 | `yolo26_pipeline/model/best.pt`, trained on WGISD | `data/runs/yolo26_<name>/weights/best.pt` |
| YOLO11 | `yolo11_pipeline/model/best.pt`, trained on WGISD | `data/runs/yolo11_<name>/weights/best.pt` |
| SAM 3 | Meta's release, also scored as released | `sam3_pipeline/model/sam3_<name>.pt` |
| SAM 3.1 | Meta's release, also scored as released | `sam3_1_pipeline/model/sam3_1_<name>.pt` |

Every model learns from the labelled `train` clips. The last job scores all of them on the `val` clips, next to the models that were not fine-tuned: `data/results/leaderboard_val.csv`. Each row says what the model is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<name>)`.

**Ship a winner.** Copy it into its pipeline's `model/` folder (a YOLO model over `model/best.pt`) and fill in that folder's README. Step 5 counts with the models in the `model/` folders.

## 5. Count the clusters in every row video

```bash
bash hpcc/count_rows.sh
```

counts every video under `data/videos`; `bash hpcc/count_rows.sh 2026/Grape--DJIOsmo` one folder under it, or one video. One job per pipeline, each with the model it ships:

| Pipeline | How it counts |
|---|---|
| SAM 2.1 | YOLO26 finds the clusters, SAM 2.1 outlines and follows each one. A cluster keeps one id for as long as it is in view. |
| YOLO26, YOLO11 | Tracker ids (ByteTrack). An id counts once it has been followed for a second. |
| SAM 3, SAM 3.1 | The model's own ids, from the text prompt "grape cluster". |

The totals land in `data/results/`:

- `cluster_counts.csv`: one line per video and model.
- `cluster_counts_by_video.csv`: one row per video, one column per model. `python fruit.py counts` in `sam2_pipeline/` prints it.
- `sam2/`, `yolo26/`, `yolo11/`, `sam3/`, `sam3_1/`: each pipeline's overlay video and per-frame counts.

A video that has its total is skipped, so run the script again when new video arrives or a job ran out of time. A total is the number of clusters a model followed, not a hand count: count one row by hand and compare before relying on the numbers.

## Single jobs

Each pipeline folder has `hpcc/run.sb`, submitted from that folder. Logs are `<job name>-<job id>.out` there.

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v2 --steps 5000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --epochs 100 --name yolo26_grapes_v2
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4 --detector runs/yolo26_grapes_v2/weights/best.pt
sbatch hpcc/run.sb check videos/row3.mp4

cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
sbatch --time=08:00:00 hpcc/run.sb finetune.py --run grapes_v2
sbatch hpcc/run.sb run.py clips

cd ~/Capstone-MSU_Fruit/yolo26_pipeline
sbatch hpcc/run.sb train.py --epochs 50 --batch 16
```

The `sam3_1_pipeline` lines work the same in `sam3_pipeline`, the `yolo26_pipeline` line in `yolo11_pipeline`. A shell command that is not a job (`fruit.py status`, `clips`, `counts`) needs the environment first: `cd sam2_pipeline && source hpcc/activate.sh`.

## Getting a better model

- **Labels decide everything.** More labelled clips from more videos (rows, light, cameras) beat more training steps on the same clips.
- **SAM 2.1:** start at 3000 steps; then `--lr`, `--encoder yes` (trains the image backbone too; needs a 40 GB GPU) and `--init` (continue from an earlier run). `python fruit.py train --help`.
- **YOLO on our clips:** `train-detector` trains at 1280 px for 100 epochs; next try `--epochs 200`, or a bigger base (`--base yolo26s-seg.pt`, fetched first on a development node: `python fruit.py download --yolo yolo26s-seg.pt`).
- **SAM 3 and SAM 3.1:** `finetune.py --epochs 40`, or `--resolution 768` if the GPU runs out of memory.
- **The video pipeline with both fine-tuned:** `python fruit.py video <video> --detector runs/yolo26_<name>/weights/best.pt --weights models/sam2.1_base_plus_<name>.pt`. To score that pair on the leaderboard, switch on the commented entry at the end of `sam2_pipeline/configs/lineup.yaml`.
