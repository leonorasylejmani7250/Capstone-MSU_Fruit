# data/ — everything the pipelines read and write

One folder for all three pipelines (`sam2_pipeline/`, `sam3_pipeline/`, `yolo26_pipeline/`): the code lives in the pipeline folders, the data and the results live here. For the SAM 2.1 pipeline this folder *is* its project folder, so its commands need no `--project` option inside the repository.

## Where the outputs are

Every result lands in `data/results/`: the comparison of all models at the top, one subfolder per pipeline for its own runs. Every command also prints the path it wrote.

| You ran | Look in |
|---|---|
| `sam2_pipeline`: `python fruit.py evaluate` (the leaderboard: every model on the held-out clips) | `results/leaderboard_val.csv` (one row per model), `results/leaderboard_val_details.csv` (per cluster), `results/compare_<clip>.mp4` and `.png` (the models side by side on a clip) |
| `sam2_pipeline`: `python fruit.py train --run <name>` (fine-tune SAM 2.1) | `models/sam2.1_<size>_<name>.pt` + `.json` recipe; training log and curves in `runs/<name>/` |
| `sam2_pipeline`: `python fruit.py train-detector` (fine-tune YOLO26 on our labels) | `runs/<name>/weights/best.pt`, `runs/<name>/results.csv`; the exported labels in `yolo_dataset/` |
| `sam2_pipeline`: `python fruit.py video <file>` (detector → SAM 2.1 on a video) | `results/sam2/overlay_<video>_<start>s.mp4`, `.counts.csv` (objects per frame), `.tracks.csv` (one row per tracked cluster) |
| `sam2_pipeline`: `demo.ipynb` / `count_test.ipynb` | `demo/results/` and `count_test/results/` (each is its own small project folder) |
| `sam3_pipeline`: `python run_sam3.py clips` | `predictions/sam3_text/<clip>/*.png` (then the leaderboard scores it) |
| `sam3_pipeline`: `python run_sam3.py video <file>` | `results/sam3/<video>_sam3.mp4` and `.csv` |
| `yolo26_pipeline`: `python train.py` | `runs/yolo26/<name>/weights/best.pt`, `results.csv`, `results.png`, PR curves, confusion matrix |
| `yolo26_pipeline`: `python evaluate.py` | printed mAP; plots in `runs/yolo26/eval_<split>_<model>/`; with several models, the table `results/yolo26/evaluate_<split>.csv` |
| `yolo26_pipeline`: `python count_test.py` | `results/yolo26/count_test.csv` and the annotated photos in `results/yolo26/count_test/` |
| `yolo26_pipeline`: `python video.py --video <file>` / `--video-dir <folder>` | `results/yolo26/<video>_detect.mp4` (or `_track`) and the per-frame `.csv` next to it; a folder's videos go under `results/yolo26/<folder name>/` |
| `yolo26_pipeline`: `python preview_labels.py` / `extract_frames.py` | `results/yolo26/label_preview/` / `frames_to_label/<video>/` |
| the HPCC job chain `hpcc/run_all.sh` | all of the above, in the same places; job logs in `sam2_pipeline/fruit-<jobid>.out` and `sam3_pipeline/sam3-<jobid>.out` |

The models the team *ships* are copied out of here into `sam2_pipeline/model/` and `yolo26_pipeline/model/` (each has a README with the recipe and the scores).

| Folder | What it holds | In git? |
|---|---|---|
| `wgisd/` | The public Embrapa WGISD grape dataset (300 photos of 5 varieties with hand-drawn cluster masks, CC BY-NC 4.0; 380 MB). YOLO26 trains on it, the SAM 2.1 demo takes its sample photos from it. The dataset's full-resolution originals and video demo frames (1 GB, used by nothing here) are left out; they are at [github.com/thsant/wgisd](https://github.com/thsant/wgisd). | yes |
| `wgisd_yolo/` | WGISD converted to YOLO segmentation labels: `images/` and `labels/`, each with `train/` (94), `val/` (16) and `test/` (27). Made by `yolo26_pipeline/prepare_wgisd.py`. | yes |
| `results/` | What the scripts write (table above): the leaderboard and comparison videos at the top, `sam2/`, `sam3/` and `yolo26/` for each pipeline's own video runs and counts. In git: `yolo11/` (the October YOLO11 baseline: its weights, training curves and metrics, the per-frame counts of its `video.py` runs on the partner's DJI videos in `CodyKresta_1025/`, and a 300-frame test run) and `yolo26/CodyKresta_1025/` (the per-frame counts of the first YOLO26 run on the same DJI videos, 3 November 2026). `yolo26_pipeline/count_test.py --model ../data/results/yolo11/best.pt` reproduces the YOLO11 count test. To keep another result, copy its numbers into a README, or add the file with `git add -f`. | `yolo11/` only |
| `videos/` | The partner's vineyard video (GoPro, phone, DJI). Too big for git, and it belongs to National Grape: copy it from the team drive. The SAM 2.1 pipeline cuts clips from here (`fruit.py clips`); `yolo26_pipeline/video.py --video data/videos/<file>` runs YOLO26 on one. | no |
| `dataset/` | Our labelled clips, written by the SAM 2.1 labeller: `JPEGImages/<clip>/` frames, `Annotations/<clip>/` masks (pixel value = cluster number), `clips.json` (train/val split, source video, draft/done). Every model is scored on the `val` clips here. Back it up to the team drive; it is the team's own labelling work. | no |
| `predictions/<model>/` | What each model drew on the clips, in the same format as `dataset/Annotations/`. `sam3_pipeline/run_sam3.py` writes `predictions/sam3_text/`; the SAM 2.1 leaderboard reads any folder here. | no |
| `models/` | Fine-tuned SAM 2.1 weights from training runs, with a `.json` recipe each. The one the team ships is copied to `sam2_pipeline/model/`. | no |
| `runs/` | Training logs and checkpoints: `runs/yolo26/<name>/` from `yolo26_pipeline/train.py`, `runs/<run>/` from SAM 2.1 fine-tuning, `runs/<name>/weights/best.pt` from `fruit.py train-detector` (scored automatically by the leaderboard). | no |
| `yolo_dataset/` | Our labelled clips exported as a YOLO dataset (`fruit.py export-yolo`, also made by `train-detector`), for `yolo26_pipeline/train.py --data`. | no |
| `demo/` | What `sam2_pipeline/demo.ipynb` makes. Delete it to run the demo from scratch. | no |
| `count_test/` | What `sam2_pipeline/count_test.ipynb` makes: one-frame clips of the 27 WGISD test photos, the count table and metrics, overlay images. | no |
| `frames_to_label/` | Still frames cut from videos by `yolo26_pipeline/extract_frames.py`, for labelling. | no |

`.gitignore` at the repository root keeps the "no" folders out of git. The annotated result videos of the October YOLO11 runs (9.5 GB) are only on the `Moe` branch of the old repository, in Git LFS.

## Getting the partner's video

Download it from the team's shared drive into `data/videos/` (any number of files; `.mp4`, `.mov`, `.avi`, `.mkv`). Never commit it. On the MSU HPCC the same folder is `~/Capstone-MSU_Fruit/data/videos/`; `sam2_pipeline/README.md` lists the ways to get files there (OnDemand upload, Globus, rsync).

**Data that arrives on the HPCC by itself** (a SharePoint or OneDrive folder synced with rclone, a research space): leave it where it lands and link it in, so nothing is copied twice and new files show up as they arrive:

```bash
cd ~/Capstone-MSU_Fruit
rmdir data/videos
ln -sT /path/to/the/synced/videos data/videos
```

(`rmdir` removes the empty `data/videos` that every command creates, and refuses if it holds files; `-T` makes `ln` fail instead of creating the link inside an existing folder.)

The SAM 2.1 pipeline reads `data/videos/` through the link (`fruit.py clips` cuts clips from every video there), and `yolo26_pipeline/video.py --video` and `sam3_pipeline/run_sam3.py video` take any path. If the synced folder holds labels as well (COCO or YOLO exports), import them with `fruit.py import-coco` / `import-yolo`. Alternatively, point `PROJECT_DIR` in `sam2_pipeline/hpcc/config.sh` at a project folder inside the synced space.

## Licences

WGISD: CC BY-NC 4.0 (`wgisd/LICENSE`); cite it if you publish results on it. Partner video: National Grape's, for this project only.
