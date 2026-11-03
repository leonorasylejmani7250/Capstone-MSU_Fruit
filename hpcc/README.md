# hpcc/ — every model, fine-tuned and scored, on the MSU HPCC

One command fine-tunes SAM 2.1 and YOLO26 on our labelled clips, runs SAM 3.1 on them, and scores all of them on the held-out clips, as a chain of batch jobs:

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1 3000 100
```

(run name, SAM 2.1 training steps, YOLO26 epochs). It submits four jobs through each pipeline's own `hpcc/run.sb`, the last one waiting for the other three, and prints their ids. `squeue --me` shows them; the logs are `sam2_pipeline/fruit-<jobid>.out` and `sam3_pipeline/sam3-<jobid>.out`; the leaderboard lands in `data/results/leaderboard_val.csv`, with the comparison video next to it.

| Job | What it does | Writes |
|---|---|---|
| 1 | `fruit.py train --run <name>`: fine-tunes SAM 2.1 base_plus on the labelled train clips (Meta's trainer; resumes if the 4-hour limit cuts it off and you submit again) | `data/models/sam2.1_base_plus_<name>.pt` |
| 2 | `fruit.py train-detector --base yolo26_pipeline/model/best.pt`: trains our YOLO26 further on the same labels | `data/runs/yolo26_<name>/weights/best.pt` |
| 3 | `run_sam3.py clips`: SAM 3.1 with the text prompt on the val clips (skipped when its environment is not set up) | `data/predictions/sam3_text/` |
| 4 | `fruit.py evaluate`: every model in `sam2_pipeline/configs/lineup.yaml` plus every fine-tuned model from jobs 1–3, on the same val clips, same rules | `data/results/leaderboard_val.csv` |

Ship a winner by copying it into its pipeline's `model/` folder and filling in that folder's README (`sam2_pipeline/model/README.md`, `yolo26_pipeline/model/README.md`).

## Before the first run

1. **Environments**, once each, on a development node (`ssh dev-amd20-v100`; those have internet access):
   - `cd ~/Capstone-MSU_Fruit/sam2_pipeline && bash hpcc/setup_env.sh` (SAM 2.1 + YOLO; 10–20 min)
   - `cd ~/Capstone-MSU_Fruit/sam3_pipeline && bash hpcc/setup_env.sh && hf auth login` (SAM 3.1; needs Hugging Face approval for `facebook/sam3.1`, see its README). Optional: without it, `run_all.sh` skips SAM 3.1.
2. **Data.** The clone's `data/` folder is the project folder for every job (`sam2_pipeline/hpcc/config.sh`). Videos go in `data/videos/`, and labelled clips in `data/dataset/`. If the incoming data is synced to the HPCC somewhere else (a SharePoint or OneDrive folder through rclone, a research space), link it in instead of copying:
   ```bash
   cd ~/Capstone-MSU_Fruit
   ln -s /mnt/research/<space>/vineyard_videos data/videos
   ```
   or point `PROJECT_DIR` in `sam2_pipeline/hpcc/config.sh` at a project folder that lives there. Either way, nothing in `data/` is in git.
3. **Labels.** Fine-tuning needs labelled clips marked *done*: cut clips (`source hpcc/activate.sh && python fruit.py clips`), then label in OnDemand Jupyter (`sam2_pipeline/README.md`, section 2), or import labels made elsewhere (`fruit.py import-coco` / `import-yolo`, then `approve`). `python fruit.py status` says how many train and val clips are usable.

## Running the pieces by hand

Every job in the chain is one `sbatch` line, so any of them can be re-run alone, from the pipeline's folder:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v2 --steps 5000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --epochs 100 --name yolo26_grapes_v2
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4 --seconds 60
cd ~/Capstone-MSU_Fruit/sam3_pipeline
sbatch hpcc/run.sb clips
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
sbatch hpcc/run.sb train.py --epochs 50 --batch 16
```

The last line retrains YOLO26 on WGISD from scratch (the public dataset), which is a different thing from job 2: job 2 continues from the shipped WGISD model on our own vineyard labels.

## Status

Written to ICER's documentation and tested piece by piece on a CPU (the same commands with `--project ../data/demo`, which is what `sam2_pipeline/demo.ipynb` builds), not yet run on the HPCC itself. The first time, start with a 30-minute smoke test before the full chain: `cd sam2_pipeline && sbatch --time=00:30:00 hpcc/run.sb --size tiny train --run smoke --steps 20`.
