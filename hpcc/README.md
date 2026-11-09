# hpcc/ — train and score every model on the MSU HPCC

## What gets trained on what

| Model | Starts from | Learns from | Command | Result |
|---|---|---|---|---|
| **SAM 2.1** | Meta's pretrained `base_plus` | our labelled clips (fine-tuning) | `fruit.py train --run <name>` in `sam2_pipeline/` | `data/models/sam2.1_base_plus_<name>.pt` |
| **YOLO26** | `yolo26_pipeline/model/best.pt` (trained on WGISD) | our labelled clips (fine-tuning) | `fruit.py train-detector --base ../yolo26_pipeline/model/best.pt --name yolo26_<name>` in `sam2_pipeline/` | `data/runs/yolo26_<name>/weights/best.pt` |
| **YOLO11** | `yolo11_pipeline/model/best.pt` (trained on WGISD) | our labelled clips (fine-tuning) | `fruit.py train-detector --base ../yolo11_pipeline/model/best.pt --name yolo11_<name>` in `sam2_pipeline/` | `data/runs/yolo11_<name>/weights/best.pt` |
| **SAM 3** | Meta's released weights | our labelled clips (fine-tuning of the detector); also scored zero-shot | `finetune.py --run <name>` in `sam3_pipeline/`, then `run.py clips` | `sam3_pipeline/model/sam3_<name>.pt`, masks in `data/predictions/sam3_<name>/` |
| **SAM 3.1** | Meta's released weights | the same | `finetune.py --run <name>` in `sam3_1_pipeline/`, then `run.py clips` | `sam3_1_pipeline/model/sam3_1_<name>.pt`, masks in `data/predictions/sam3_1_<name>/` |
| leaderboard | — | — | `fruit.py evaluate` in `sam2_pipeline/` | `data/results/leaderboard_val.csv` |

The leaderboard scores all of them on the same held-out clips, next to the pretrained, WGISD-only and zero-shot models. Each row says what it is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<name>)`.

## One command

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1 3000 100 20
```

Arguments: run name, SAM 2.1 training steps, YOLO epochs, SAM 3 epochs. It submits the jobs above as a chain (SAM 3 and SAM 3.1 each as three jobs: zero-shot masks, fine-tune, masks with the fine-tuned model; the leaderboard waits for everything) and prints one line per job with what it does and where its result lands. `squeue --me` shows them; logs are `sam2_pipeline/fruit-<jobid>.out`, `sam3_pipeline/sam3-<jobid>.out` and `sam3_1_pipeline/sam3_1-<jobid>.out`. A SAM 3 pipeline is submitted only when its environment exists (step 1 below); the script says so either way.

Ship a winner by copying it into its pipeline's `model/` folder and filling in that folder's README.

## Before the first run

1. **Environments**, once, on a development node (`ssh dev-amd20-v100`):
   ```bash
   cd ~/Capstone-MSU_Fruit/sam2_pipeline && bash hpcc/setup_env.sh
   cd ~/Capstone-MSU_Fruit/sam3_pipeline && bash hpcc/setup_env.sh
   cd ~/Capstone-MSU_Fruit/sam3_1_pipeline && bash hpcc/setup_env.sh
   ```
   The SAM 2.1 environment also runs the YOLO fine-tuning. Each SAM 3 setup ends by printing its two `hf` lines (log in, download the release into `model/`); they need Hugging Face approval for `facebook/sam3` and `facebook/sam3.1`.
2. **Data.** Videos go under `data/videos/`, in any folder structure. From SharePoint: Globus (`msu#onedrive` → `msuhpcc`, sync option) or `bash hpcc/sync_sharepoint.sh` with rclone; a big library lives on the research space with `data/videos` as a symlink to it. All of it in [`data/README.md`](../data/README.md), "Video from SharePoint".
3. **Labels.** Cut clips and label them (`sam2_pipeline/README.md`, section 2):
   ```bash
   cd ~/Capstone-MSU_Fruit/sam2_pipeline
   source hpcc/activate.sh
   python fruit.py clips
   ```
   then label in OnDemand Jupyter, or import labels made elsewhere (`fruit.py import-coco` / `import-yolo`, then `approve`). `python fruit.py status` shows how many clips are usable.

## Getting a better model

- **Labels decide everything.** More labelled clips from more videos (rows, light, cameras) beat more training steps on the same clips. Hold whole videos out for val (`fruit.py clips` does).
- **SAM 2.1:** start at `--steps 3000`; then `--lr`, `--encoder yes` (train the image backbone too; needs a 40 GB+ GPU) and `--init` (continue from an earlier run). See `fruit.py train --help`.
- **YOLO on our clips:** `train-detector` trains at 1280 px for 100 epochs; next try `--epochs 200`, or a bigger base (`--base yolo26s-seg.pt`; fetch it first on a dev node: `python fruit.py download --yolo yolo26s-seg.pt`). For a stronger WGISD model, see `yolo26_pipeline/README.md`, section 1.
- **SAM 3 / SAM 3.1:** `finetune.py --epochs 40`, or `--resolution 768` if the GPU runs out of memory. The text prompt is the category name the model learns (`--text`).
- **Detector → SAM 2.1 is the pipeline for video.** With a fine-tuned SAM 2.1 and a fine-tuned detector, add the combined row to `sam2_pipeline/configs/lineup.yaml` (the commented entry at the end) and run `evaluate` again.
- Ship a model only when it wins on the held-out clips, and write its numbers into `model/README.md`.

## Single jobs

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v2 --steps 5000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --epochs 100 --name yolo26_grapes_v2
sbatch hpcc/run.sb train-detector --base ../yolo11_pipeline/model/best.pt --epochs 100 --name yolo11_grapes_v2
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4 --seconds 60 --detector-weights-trained runs/yolo26_grapes_v2/weights/best.pt
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
sbatch hpcc/run.sb run.py clips --checkpoint release
sbatch --time=08:00:00 hpcc/run.sb finetune.py --run grapes_v2
sbatch hpcc/run.sb run.py clips
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
sbatch hpcc/run.sb train.py --epochs 50 --batch 16
```

The same `sam3_1_pipeline` lines work in `sam3_pipeline`. The last line retrains YOLO26 on the public WGISD dataset from Ultralytics' pretrained weights (same for `yolo11_pipeline`); the `train-detector` jobs continue from those models on our own labels.
