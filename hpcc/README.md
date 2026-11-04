# hpcc/ — train and score every model on the MSU HPCC

One command fine-tunes SAM 2.1 and YOLO26 on our labelled clips, runs SAM 3.1 on them, and scores all of them on the held-out clips:

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1 3000 100
```

Arguments: run name, SAM 2.1 training steps, YOLO26 epochs. It submits four batch jobs and prints their ids; `squeue --me` shows them, the logs are `sam2_pipeline/fruit-<jobid>.out` and `sam3_pipeline/sam3-<jobid>.out`.

| Job | Command | Writes |
|---|---|---|
| 1 | `fruit.py train --run <name>`: fine-tune SAM 2.1 on the labelled train clips | `data/models/sam2.1_base_plus_<name>.pt` |
| 2 | `fruit.py train-detector --base yolo26_pipeline/model/best.pt`: fine-tune our YOLO26 on the same labels | `data/runs/yolo26_<name>/weights/best.pt` |
| 3 | `run_sam3.py clips`: SAM 3.1 on the val clips (skipped if its environment is missing) | `data/predictions/sam3_text/` |
| 4 | `fruit.py evaluate`, after 1–3: every model in `sam2_pipeline/configs/lineup.yaml` plus the three above, on the same val clips | `data/results/leaderboard_val.csv` |

Ship a winner by copying it into its pipeline's `model/` folder and filling in that folder's README.

## Before the first run

1. **Environments**, once, on a development node (`ssh dev-amd20-v100`):
   ```bash
   cd ~/Capstone-MSU_Fruit/sam2_pipeline && bash hpcc/setup_env.sh
   cd ~/Capstone-MSU_Fruit/sam3_pipeline && bash hpcc/setup_env.sh && ~/envs/fruit_sam3/bin/hf auth login
   ```
   The second line needs Hugging Face approval for `facebook/sam3.1` (`sam3_pipeline/README.md`); skip it and `run_all.sh` skips SAM 3.1.
2. **Data.** Videos go in `data/videos/`. If they are synced to the HPCC somewhere else (SharePoint or OneDrive through rclone, a research space), link that folder in instead of copying:
   ```bash
   cd ~/Capstone-MSU_Fruit
   rmdir data/videos
   ln -sT /path/to/the/synced/videos data/videos
   ```
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
- **YOLO26 on our labels:** `train-detector` trains at 1280 px for 100 epochs from the shipped model; next try `--epochs 200`, or `--base yolo26s-seg.pt` (fetch it first on a dev node: `python fruit.py download --yolo yolo26s-seg.pt`). For a stronger WGISD model, see `yolo26_pipeline/README.md`, section 2.
- **Detector → SAM 2.1 is the pipeline for video.** With a fine-tuned SAM 2.1 and a fine-tuned detector, add the combined row to `sam2_pipeline/configs/lineup.yaml` (the commented `YOLO26n-seg · ours → SAM 2.1 fine-tuned` entry) and run `evaluate` again.
- Ship a model only when it wins on the held-out clips, and write its numbers into `model/README.md`.

## Single jobs

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v2 --steps 5000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --epochs 100 --name yolo26_grapes_v2
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4 --seconds 60 --detector-weights-trained runs/yolo26_grapes_v2/weights/best.pt
cd ~/Capstone-MSU_Fruit/sam3_pipeline
sbatch hpcc/run.sb clips
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
sbatch hpcc/run.sb train.py --epochs 50 --batch 16
```

The last line retrains YOLO26 on the public WGISD dataset; job 2 above continues from that model on our own labels.
