# data/ — everything the pipelines read and write

One folder for all the pipelines. For the SAM 2.1 pipeline this folder is the project folder, so its commands need no `--project` inside the repository.

## Where the outputs are

Everything lands in `data/results/`: the leaderboard of all models at the top, one subfolder per pipeline. Every command prints the path it wrote.

| You ran | Look in |
|---|---|
| `sam2_pipeline`: `python fruit.py evaluate` | `results/leaderboard_val.csv` (one row per model), `results/leaderboard_val_details.csv` (per cluster), `results/compare_<clip>.mp4` and `.png` |
| `sam2_pipeline`: `python fruit.py train --run <name>` | `models/sam2.1_<size>_<name>.pt` (+ `.json` recipe), log and curves in `runs/<name>/` |
| `sam2_pipeline`: `python fruit.py train-detector --name <name>` | `runs/<name>/weights/best.pt`, `runs/<name>/recipe.json` (what it started from), `results.csv`; the exported labels in `yolo_dataset/` |
| `sam2_pipeline`: `python fruit.py video <file>` | `results/sam2/overlay_<video>_<start>s.mp4`, `.counts.csv`, `.tracks.csv` |
| `sam2_pipeline`: `demo.ipynb`, `count_test.ipynb` | `demo/results/`, `count_test/results/` |
| `sam3_pipeline` / `sam3_1_pipeline`: `python run.py clips` | `predictions/sam3_text/` or `predictions/sam3_1_text/` (Meta's release), `predictions/<pipeline>_<run>/` (a fine-tuned model): one PNG per frame plus `recipe.json`; the leaderboard scores every such folder |
| `sam3_pipeline` / `sam3_1_pipeline`: `python finetune.py --run <name>` | the COCO dataset in `sam3_dataset/`, log and checkpoint in `runs/<pipeline>_<name>/`, the model in the pipeline's `model/` |
| `sam3_pipeline` / `sam3_1_pipeline`: `python run.py video <file>` | `results/sam3/<video>_sam3.mp4` or `results/sam3_1/<video>_sam3_1.mp4`, and `.csv` |
| `yolo26_pipeline` / `yolo11_pipeline`: `python train.py --name <name>` | `runs/yolo26/<name>/` or `runs/yolo11/<name>/`: `weights/best.pt`, `results.csv`, `results.png`, curves, confusion matrix |
| `yolo26_pipeline` / `yolo11_pipeline`: `python evaluate.py` | printed mAP; plots in `runs/<pipeline>/eval_<split>_<name>/`; with several models, `results/<pipeline>/evaluate_<split>.csv` |
| `yolo26_pipeline` / `yolo11_pipeline`: `python count_test.py` | `results/<pipeline>/count_test.csv` and the annotated photos in `results/<pipeline>/count_test/` |
| `yolo26_pipeline` / `yolo11_pipeline`: `python video.py --video <file>` / `--video-dir <folder>` | `results/<pipeline>/<video>_<mode>.mp4` and `.csv` (a folder's videos under `results/<pipeline>/<folder>/`) |
| `wgisd_yolo/prepare_wgisd.py --preview 8` | `wgisd_yolo/preview/` |
| `sam2_pipeline`: `python fruit.py clips` | `dataset/JPEGImages/<clip>/`, `dataset/clips.json`, `dataset/videos.csv` |
| `hpcc/sync_sharepoint.sh` | new videos under `videos/`, `videos/.sync.log`, then the same as `fruit.py clips` |
| `hpcc/run_all.sh` | the same places; job logs in `sam2_pipeline/fruit-<jobid>.out`, `sam3_pipeline/sam3-<jobid>.out`, `sam3_1_pipeline/sam3_1-<jobid>.out` |

The models the team ships are copied from here into each pipeline's `model/` folder.

## Folders

| Folder | What it holds | In git |
|---|---|---|
| `wgisd/` | Public Embrapa WGISD grape dataset: 300 photos, hand-drawn cluster masks (CC BY-NC 4.0). | yes |
| `wgisd_yolo/` | WGISD as YOLO labels for the YOLO pipelines: `images/`, `labels/`, split `train/` 94, `val/` 16, `test/` 27; `grape.yaml` describes it; `prepare_wgisd.py` rebuilds it from `wgisd/` (`--preview 8` draws the labels on a few photos). | yes |
| `results/` | Outputs (table above). In git: `yolo11/CodyKresta_1025/` and `yolo26/CodyKresta_1025/`, the per-frame counts of both YOLO models on the partner's DJI videos. | those two |
| `videos/` | The partner's video, any folder structure (synced from SharePoint: section below); never committed. | no |
| `dataset/` | Our labelled clips: `JPEGImages/<clip>/` frames, `Annotations/<clip>/` masks (pixel value = cluster number), `clips.json` (split, source video, draft/done), `videos.csv` (every video and the clips cut from it). Back it up to the team drive. | no |
| `predictions/<model>/` | What each model drew on the clips, same format as `Annotations/`. The SAM 3 pipelines write theirs here with a `recipe.json`. | no |
| `models/` | Fine-tuned SAM 2.1 weights (+ `.json` recipe each). | no |
| `runs/` | Training runs: `runs/yolo26/<name>/` and `runs/yolo11/<name>/` (YOLO on WGISD), `runs/<name>/` (SAM 2.1 fine-tuning, and `train-detector`'s `weights/best.pt` + `recipe.json`), `runs/sam3_<name>/` and `runs/sam3_1_<name>/` (SAM 3 fine-tuning). In git: `runs/yolo11/wgisd/`, the October YOLO11 training run (curves and metrics). | `yolo11/wgisd/` |
| `yolo_dataset/` | Our labelled clips exported as a YOLO dataset (`fruit.py export-yolo`), what `train-detector` and `train.py --data` learn from. | no |
| `sam3_dataset/` | Our labelled clips exported as a COCO dataset by `finetune.py` (train clips → `train/`, val clips → `test/`), what the SAM 3 trainer learns from. | no |
| `demo/`, `count_test/` | What the two SAM 2.1 notebooks make. Delete to start over. | no |

## Video from SharePoint

The pipelines read every video under `data/videos/`, at any depth and with any name: the SharePoint library can land here exactly as it is. Hidden files, Office and macOS sidecars (`._*`, `~$*`), partial downloads and GoPro `.LRV`/`.THM` proxies are ignored, a file written in the last two minutes is left for the next run (it may still be syncing), and an unreadable file is reported and skipped. Clips are named after the video's path (`CodyKresta_1025__DJI_0001_t0003200`), so two cameras' `DJI_0001.MP4` in different folders never collide; `data/dataset/videos.csv` lists every video with the clips cut from it.

**Where the files live.** Home has a 100 GB quota, so a library of any size goes on the group's research space (or `$SCRATCH`, which deletes files untouched for 45 days; SharePoint stays the master copy either way) with `data/videos` pointing at it:

```bash
cd ~/Capstone-MSU_Fruit
mkdir -p /mnt/research/<group>/fruit_videos
rmdir data/videos
ln -sT /mnt/research/<group>/fruit_videos data/videos
```

**Route A, Globus (no software, MSU's own OneDrive connector).** At <https://app.globus.org/file-manager>, two panels: left `msu#onedrive` (log in with the MSU address; the root shows *My files*, *Shared* and *Shared libraries*, where the SharePoint library is), right `msuhpcc` at `~/Capstone-MSU_Fruit/data/videos` (or the research-space folder). Select the library's video folder, *Transfer & Sync Options* ▸ *sync - only transfer new or changed files*, *Start*. Run the same transfer again whenever new video has been uploaded, or set it to repeat on a schedule in the same options. Then, on a development node:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py clips
```

**Route B, rclone (scriptable).** Once, on a development node with `ssh -X`: `module load rclone`, `rclone config` (new remote `sharepoint`, type `onedrive`, auto config `y`, log in in the browser that opens on the node, then *search* for the team's SharePoint site and pick its document library), and `export FRUIT_REMOTE="sharepoint:Shared Documents/<video folder>"` in `~/.bashrc`. Then, whenever new video has been uploaded:

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/sync_sharepoint.sh
```

It copies new videos only (never deletes anything local), logs to `data/videos/.sync.log`, and cuts clips from what is new. If `rclone config` ends in an `AADSTS` error, MSU's Microsoft tenant does not allow rclone: use Route A.

**Then** label the new clips (`sam2_pipeline/README.md`, section 2) and train (`hpcc/README.md`). Clips, labels and splits are never redone: a second run cuts clips only from new videos, and a video already split stays on its side. Labels made elsewhere (COCO or YOLO exports) come in with `fruit.py import-coco` / `import-yolo`. A single video from a laptop: copy it into `data/videos/` by OnDemand (*Files ▸ Upload*, under 1 GB) or `rsync -av <file> <netid>@rsync.hpcc.msu.edu:~/Capstone-MSU_Fruit/data/videos/`.

## Licences

WGISD: CC BY-NC 4.0 (`wgisd/LICENSE`). Partner video: National Grape's, for this project only.
