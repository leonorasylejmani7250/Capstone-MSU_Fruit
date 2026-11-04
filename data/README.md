# data/ — everything the pipelines read and write

One folder for all three pipelines. For the SAM 2.1 pipeline this folder is the project folder, so its commands need no `--project` inside the repository.

## Where the outputs are

Everything lands in `data/results/`: the leaderboard of all models at the top, one subfolder per pipeline. Every command prints the path it wrote.

| You ran | Look in |
|---|---|
| `sam2_pipeline`: `python fruit.py evaluate` | `results/leaderboard_val.csv` (one row per model), `results/leaderboard_val_details.csv` (per cluster), `results/compare_<clip>.mp4` and `.png` |
| `sam2_pipeline`: `python fruit.py train --run <name>` | `models/sam2.1_<size>_<name>.pt` (+ `.json` recipe), log and curves in `runs/<name>/` |
| `sam2_pipeline`: `python fruit.py train-detector --name <name>` | `runs/<name>/weights/best.pt`, `runs/<name>/results.csv`; the exported labels in `yolo_dataset/` |
| `sam2_pipeline`: `python fruit.py video <file>` | `results/sam2/overlay_<video>_<start>s.mp4`, `.counts.csv`, `.tracks.csv` |
| `sam2_pipeline`: `demo.ipynb`, `count_test.ipynb` | `demo/results/`, `count_test/results/` |
| `sam3_pipeline`: `python run_sam3.py clips` | `predictions/sam3_text/<clip>/*.png` (the leaderboard scores it) |
| `sam3_pipeline`: `python run_sam3.py video <file>` | `results/sam3/<video>_sam3.mp4` and `.csv` |
| `yolo26_pipeline`: `python train.py --name <name>` | `runs/yolo26/<name>/weights/best.pt`, `results.csv`, `results.png`, curves, confusion matrix |
| `yolo26_pipeline`: `python evaluate.py` | printed mAP; plots in `runs/yolo26/eval_<split>_<model>/`; with several models, `results/yolo26/evaluate_<split>.csv` |
| `yolo26_pipeline`: `python count_test.py` | `results/yolo26/count_test.csv` and the annotated photos in `results/yolo26/count_test/` |
| `yolo26_pipeline`: `python video.py --video <file>` / `--video-dir <folder>` | `results/yolo26/<video>_detect.mp4` and `.csv` (a folder's videos under `results/yolo26/<folder>/`) |
| `yolo26_pipeline`: `python preview_labels.py` / `extract_frames.py` | `results/yolo26/label_preview/` / `frames_to_label/<video>/` |
| `hpcc/run_all.sh` | the same places; job logs in `sam2_pipeline/fruit-<jobid>.out` and `sam3_pipeline/sam3-<jobid>.out` |

The models the team ships are copied from here into `sam2_pipeline/model/` and `yolo26_pipeline/model/`.

## Folders

| Folder | What it holds | In git |
|---|---|---|
| `wgisd/` | Public Embrapa WGISD grape dataset: 300 photos, hand-drawn cluster masks (CC BY-NC 4.0). | yes |
| `wgisd_yolo/` | WGISD as YOLO labels: `images/` and `labels/`, split `train/` 94, `val/` 16, `test/` 27 (`yolo26_pipeline/prepare_wgisd.py`). | yes |
| `results/` | Outputs (table above). In git: `yolo11/` (the October YOLO11 baseline: weights, curves, metrics, per-frame counts on the DJI videos) and `yolo26/CodyKresta_1025/` (YOLO26 per-frame counts on the same videos). | those two |
| `videos/` | The partner's video. From the team drive; never committed. | no |
| `dataset/` | Our labelled clips: `JPEGImages/<clip>/` frames, `Annotations/<clip>/` masks (pixel value = cluster number), `clips.json` (split, source video, draft/done). Back it up to the team drive. | no |
| `predictions/<model>/` | What each model drew on the clips, same format as `Annotations/`. | no |
| `models/` | Fine-tuned SAM 2.1 weights (+ `.json` recipe each). | no |
| `runs/` | Training runs: `runs/yolo26/<name>/` (YOLO26 on WGISD), `runs/<name>/` (SAM 2.1 fine-tuning, and `train-detector`'s `weights/best.pt`). | no |
| `yolo_dataset/` | Our labelled clips exported as a YOLO dataset (`fruit.py export-yolo`). | no |
| `demo/`, `count_test/` | What the two SAM 2.1 notebooks make. Delete to start over. | no |
| `frames_to_label/` | Still frames from videos (`yolo26_pipeline/extract_frames.py`). | no |

## Getting the partner's video

Copy it from the team drive into `data/videos/` (`.mp4`, `.mov`, `.avi`, `.mkv`). On the HPCC the folder is `~/Capstone-MSU_Fruit/data/videos/` (`sam2_pipeline/README.md` lists the ways to get files there).

If the video is synced to the HPCC somewhere else (SharePoint or OneDrive through rclone, a research space), link that folder in instead of copying:

```bash
cd ~/Capstone-MSU_Fruit
rmdir data/videos
ln -sT /path/to/the/synced/videos data/videos
```

New files then show up as they arrive. Labels made elsewhere (COCO or YOLO exports) come in with `fruit.py import-coco` / `import-yolo`.

## Licences

WGISD: CC BY-NC 4.0 (`wgisd/LICENSE`). Partner video: National Grape's, for this project only.
