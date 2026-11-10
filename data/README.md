# data/ — everything the pipelines read and write

One folder for all the pipelines. For the SAM 2.1 pipeline it is the project folder, so its commands need no `--project` inside the repository.

## Folders

| Folder | What it holds | In git |
|---|---|---|
| `videos/` | The partner's video, in any folder structure. On the HPCC a link to the team's shared folder (section below). | no |
| `dataset/` | Our clips and their labels: `JPEGImages/<clip>/` frames, `Annotations/<clip>/` masks (pixel value = cluster number), `clips.json` (split, source video, draft or done, who labels it). On the HPCC a link to the team's shared folder. | no |
| `results/` | Every output (next table). | `yolo11/CodyKresta_1025/`, `yolo26/CodyKresta_1025/` |
| `predictions/<model>/` | What each model drew on the clips, in the format of `Annotations/`; the leaderboard scores these. | no |
| `models/` | Fine-tuned SAM 2.1 weights, each with a `.json` recipe. | no |
| `runs/` | Training runs: `runs/<name>/` (SAM 2.1 fine-tuning and `train-detector`), `runs/yolo26/<name>/` and `runs/yolo11/<name>/` (YOLO on WGISD), `runs/sam3_<name>/` and `runs/sam3_1_<name>/`. | `yolo11/wgisd/` |
| `wgisd/` | Public Embrapa WGISD grape dataset: 300 photos with hand-drawn cluster masks (CC BY-NC 4.0). | yes |
| `wgisd_yolo/` | WGISD as YOLO labels (`train/` 94 photos, `val/` 16, `test/` 27), `grape.yaml`, and `prepare_wgisd.py`, which rebuilds it. | yes |
| `yolo_dataset/`, `sam3_dataset/` | Our labelled clips exported for YOLO and for SAM 3 training; written by the training commands. | no |
| `demo/`, `count_test/` | What the two SAM 2.1 notebooks make. Delete to start over. | no |

## Where the outputs are

Everything lands in `results/`. Every command prints the path it wrote.

| Output | File |
|---|---|
| The leaderboard: every model on our labelled val clips | `results/leaderboard_val.csv`, per cluster in `leaderboard_val_details.csv`, side by side in `compare_<clip>.mp4` |
| Cluster totals: every model on every row video | `results/cluster_counts.csv` (one line per video and model), `cluster_counts_by_video.csv` (one row per video) |
| A pipeline's overlay videos and per-frame counts | `results/sam2/`, `results/yolo26/`, `results/yolo11/`, `results/sam3/`, `results/sam3_1/` |
| What every detector sees in one video (`fruit.py check`) | `results/check/<video>/` |
| YOLO on the WGISD test photos | `results/yolo26/count_test.csv`, `count_test/`, `evaluate_<split>.csv` (the same under `yolo11/`) |
| Fine-tuned models | SAM 2.1: `models/`. YOLO: `runs/<name>/weights/best.pt`. SAM 3 and SAM 3.1: the pipeline's `model/` folder. |

The models the team ships are copied from here into each pipeline's `model/` folder.

## Video from SharePoint

The video lives once, in the team's folder on the research space, and every clone links to it. `bash hpcc/setup.sh` makes the link ([`hpcc/README.md`](../hpcc/README.md)).

To bring in new video:

1. **OneDrive on the web.** Open the shared SharePoint folder, select the new videos, *Copy to* a folder of your own such as *My files / FRUIT_videos*. Globus cannot open a folder that is only shared with you.
2. **Globus**, <https://app.globus.org/file-manager>. Left panel: `msu#onedrive`, *My files / FRUIT_videos*, select all. Right panel: `msuhpcc`, path `/mnt/ufs18/rs/MSU-Fruit/videos/`. In *Transfer & Timer Options* tick *sync - only transfer new or changed files* (*modification time is newer*) and *preserve source file modification times*; leave *delete files on destination* off. *Start*.
3. **On the HPCC**, when the transfer is done:

   ```bash
   chmod -R g+rwX /mnt/research/MSU-Fruit/videos
   cd ~/Capstone-MSU_Fruit/sam2_pipeline
   source hpcc/activate.sh
   python fruit.py clips
   ```

Any folder structure and any file names work: every video under `videos/` is read, at any depth. Clips are named after the video's path, so two cameras' `DJI_0001.MP4` in different folders do not collide. A second `fruit.py clips` cuts clips only from new videos and never moves a clip between train and val. Footage that should not be used goes into a folder whose name starts with a dot, such as `videos/.unused/`.

A single video from a laptop: OnDemand (*Files ▸ Upload*, under 1 GB) or `rsync -av <file> <netid>@rsync.hpcc.msu.edu:/mnt/research/MSU-Fruit/videos/`.

## Licences

WGISD: CC BY-NC 4.0 (`wgisd/LICENSE`). Partner video: National Grape's, for this project only.
