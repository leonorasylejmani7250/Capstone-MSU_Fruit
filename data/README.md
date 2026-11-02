# data/ — everything the pipelines read and write

One folder for all three pipelines (`sam2_pipeline/`, `sam3_pipeline/`, `yolo26_pipeline/`): the code lives in the pipeline folders, the data and the results live here. For the SAM 2.1 pipeline this folder *is* its project folder, so its commands need no `--project` option inside the repository.

| Folder | What it holds | In git? |
|---|---|---|
| `wgisd/` | The public Embrapa WGISD grape dataset (300 photos of 5 varieties with hand-drawn cluster masks, CC BY-NC 4.0; 380 MB). YOLO26 trains on it, the SAM 2.1 demo takes its sample photos from it. The dataset's full-resolution originals and video demo frames (1 GB, used by nothing here) are left out; they are at [github.com/thsant/wgisd](https://github.com/thsant/wgisd). | yes |
| `wgisd_yolo/` | WGISD converted to YOLO segmentation labels: `images/` and `labels/`, each with `train/` (94), `val/` (16) and `test/` (27). Made by `yolo26_pipeline/prepare_wgisd.py`. | yes |
| `results/` | What the scripts write: `leaderboard_*.csv` and comparison videos from the SAM 2.1 pipeline, `yolo26/` (YOLO26 counts and annotated videos), `sam3/`, `label_preview/`. Only `yolo11/` is in git: the October YOLO11 baseline (its weights, curves, metrics and the per-frame counts on the partner's videos). To keep another result, copy its numbers into a README, or add the file with `git add -f`. | `yolo11/` only |
| `videos/` | The partner's vineyard video (GoPro, phone, DJI). Too big for git, and it belongs to National Grape: copy it from the team drive. The SAM 2.1 pipeline cuts clips from here (`fruit.py clips`); `yolo26_pipeline/video.py --video data/videos/<file>` runs YOLO26 on one. | no |
| `dataset/` | Our labelled clips, written by the SAM 2.1 labeller: `JPEGImages/<clip>/` frames, `Annotations/<clip>/` masks (pixel value = cluster number), `clips.json` (train/val split, source video, draft/done). Every model is scored on the `val` clips here. Back it up to the team drive; it is the team's own labelling work. | no |
| `predictions/<model>/` | What each model drew on the clips, in the same format as `dataset/Annotations/`. `sam3_pipeline/run_sam3.py` writes `predictions/sam3_text/`; the SAM 2.1 leaderboard reads any folder here. | no |
| `models/` | Fine-tuned SAM 2.1 weights from training runs, with a `.json` recipe each. The one the team ships is copied to `sam2_pipeline/model/`. | no |
| `runs/` | Training logs and checkpoints: `runs/yolo26/<name>/` from `yolo26_pipeline/train.py`, `runs/<run>/` from SAM 2.1 fine-tuning. | no |
| `demo/` | What `sam2_pipeline/demo.ipynb` makes. Delete it to run the demo from scratch. | no |
| `frames_to_label/` | Still frames cut from videos by `yolo26_pipeline/extract_frames.py`, for labelling. | no |

`.gitignore` at the repository root keeps the "no" folders out of git. The annotated result videos of the October YOLO11 runs (9.5 GB) are only on the `Moe` branch of the old repository, in Git LFS.

## Getting the partner's video

Download it from the team's shared drive into `data/videos/` (any number of files; `.mp4`, `.mov`, `.avi`, `.mkv`). Never commit it. On the MSU HPCC the same folder is `~/Capstone-MSU_Fruit/data/videos/`; `sam2_pipeline/README.md` lists the ways to get files there (OnDemand upload, Globus, rsync).

## Licences

WGISD: CC BY-NC 4.0 (`wgisd/LICENSE`); cite it if you publish results on it. Partner video: National Grape's, for this project only.
