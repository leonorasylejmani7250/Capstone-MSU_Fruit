# SAM 3 pipeline

SAM 3 (Meta, November 2025) segments and tracks every object that matches a text prompt ("grape cluster"), with no boxes drawn by anyone. SAM 3.1 (`../sam3_1_pipeline/`) is the same model family with a faster multi-object tracker; this folder keeps the original so the two can be compared on the leaderboard. This folder runs it on our clips and videos, fine-tunes it on our labelled clips, and writes its masks where the leaderboard in `../sam2_pipeline` scores them next to every other model. It reads everything from `../data/` and writes to `../data/predictions/` (masks for the leaderboard) and `../data/results/sam3/` (videos and counts). The weights it uses live in `model/`.

```
../data/dataset (labelled clips) ──▶ run.py clips ──▶ ../data/predictions/sam3_text/ ──▶ ../sam2_pipeline: python fruit.py evaluate
../data/dataset (labelled clips) ──▶ finetune.py ──▶ model/sam3_<run>.pt ──▶ run.py clips ──▶ ../data/predictions/sam3_<run>/ ──▶ evaluate
../data/videos/<file>            ──▶ run.py video ──▶ ../data/results/sam3/<file>_sam3.mp4 + .csv, the total in ../data/results/cluster_counts.csv
```

| File | What it does |
|---|---|
| `run.py` | `clips`: masks for the labelled val clips. `video`: counts the clusters in a video, or in every video of a folder. |
| `finetune.py` | Fine-tunes the model on our labelled clips with Meta's trainer and writes the result to `model/`. |
| `finetune.yaml` | Meta's fine-tuning recipe with our dataset and settings (filled in by `finetune.py`). |
| `meta_trainer.py` | Starts Meta's trainer for `finetune.py`, with the one correction it needs to train. |
| `pipeline.py` | The names every script uses: model version, weights, data folder, where predictions and results go. |
| `model/` | The weights: Meta's release and our fine-tuned checkpoints ([model/README.md](model/README.md)). |
| `hpcc/` | HPCC environment (`setup_env.sh` builds it, `activate.sh` turns it on in a terminal) and batch job (`run.sb`). |
| `requirements.txt` | Python packages (Meta's `sam3` package with its trainer). |

## 1. What you need

- An NVIDIA GPU with CUDA 12.6 or newer (the HPCC, Google Colab, a lab machine). Not a CPU, not a Mac.
- Python 3.12 or newer, PyTorch 2.7 or newer.
- Access to the weights: request it at [facebook/sam3](https://huggingface.co/facebook/sam3) on Hugging Face, wait for the approval email, then log in once on the machine that runs it: `hf auth login`.
- Labelled clips in `../data/dataset/` from the SAM 2.1 labeller (`clips` and `finetune.py`), or a video in `../data/videos/` (`video`).

## 2. Install

**On the HPCC**, once, on a development node:

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
bash hpcc/setup_env.sh
```

Then run the `hf auth login` and `hf download` lines it prints: they put Meta's weights in `model/`, where a compute node without internet finds them. After that, in every new terminal:

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
```

**On another Linux computer with an NVIDIA GPU.** Not on the HPCC, where `python` is the cluster's own Python 3.11 with the cluster's packages:

```bash
cd Capstone-MSU_Fruit/sam3_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
hf auth login
hf download facebook/sam3 sam3.pt --local-dir model
```

Use the CUDA build of PyTorch that matches the machine (`cu126`, or `cu128` on newer cards). The last line keeps Meta's weights in `model/`.

Colab: *Runtime ▸ Change runtime type ▸ T4 GPU*, then `!git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit/sam3_pipeline && pip install -q -r requirements.txt`, then `from huggingface_hub import login; login()`. Copy `data/dataset/` from Drive into the clone's `data/`.

## 3. Run

Masks for the held-out clips, then the leaderboard:

```bash
python run.py clips
cd ../sam2_pipeline && python fruit.py evaluate
```

`run.py` uses the newest `.pt` in `model/`: Meta's release after the install above, a fine-tuned model after `finetune.py`. `--checkpoint release` forces Meta's weights, `--checkpoint <file>` a particular file. Each run writes `../data/predictions/<name>/` with a `recipe.json`, and `fruit.py evaluate` adds a row for it by itself: `SAM 3 · text 'grape cluster' · zero-shot` for the release, `SAM 3 · text 'grape cluster' → fine-tuned on our clips (<run>)` for a fine-tuned model.

Count the clusters in a video, or in every video of a folder:

```bash
python run.py video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
python run.py video ../data/videos/2026/Grape--DJIOsmo
```

The video goes through the model in stretches of 10 seconds that share one frame; a cluster keeps its id from one stretch to the next when its masks on the shared frame overlap. The total is the number of ids seen in at least two frames. Writes `../data/results/sam3/<video>_sam3.mp4` and `.csv` (`frame`, `visible_now`, `clusters_so_far`), and adds the video's total to `../data/results/cluster_counts.csv`, the file every pipeline adds to. `--start 60 --seconds 30` runs a stretch for a quick look; only a whole video adds its total. In a folder, videos that have a total are skipped unless `--overwrite`.

A total is the number of clusters the model followed, not a hand count, and not a yield estimate.

Options for both go before the command name, as in `python run.py --text "green grapes" video …`: `--text` (one phrase per prompt), `--prob 0.4` (output threshold), `--data <folder>` (project folder; default `../data` or `$PROJECT_DIR`).

## 4. Fine-tune on our clips

```bash
python finetune.py --run grapes_v1
```

On the HPCC: `sbatch --time=04:00:00 hpcc/run.sb finetune.py --run grapes_v1`. A job of 4 hours or less usually starts within minutes; one that runs out of time continues from its last finished epoch when the same line is submitted again.

It exports the labelled clips as a COCO dataset (`../data/sam3_dataset/`: train clips → `train/`, val clips → `test/`), runs Meta's trainer on it starting from the released weights (log, curves and checkpoint in `../data/runs/sam3_grapes_v1/`), and merges the fine-tuned detector into the released video checkpoint: `model/sam3_grapes_v1.pt` with a `.json` recipe. The trainer fine-tunes the detector, the part that finds clusters from the text; the tracker that follows them through the video stays as released.

| Argument | Default | Meaning |
|---|---|---|
| `--run` | required | Name of the run |
| `--epochs` | `20` | Training epochs; an epoch is one pass over the training images |
| `--epoch-size` | `100` | At least this many images per epoch; with fewer training images, each is shown several times |
| `--resolution` | `1008` | Input size; lower it if the GPU runs out of memory |
| `--batch` | `1` | Batch size |
| `--every` | `2` | Use every n-th frame of a clip |
| `--text` | `grape cluster` | The category name, which is the text prompt the model learns |
| `--dry-run` | | Export the dataset and write the recipe without training |
| `--merge-only <checkpoint>` | | Merge an already trained `checkpoints/checkpoint.pt` into the released weights |

With the defaults that is 2,000 training steps or more, about a second each on an A100 or H200: 40 minutes for a few labelled clips.

Then score it: `python run.py clips` (the new file is the newest in `model/`) and `python fruit.py evaluate` in `../sam2_pipeline`. The fine-tuned file is 3 GB and ignored by git; share it through a GitHub Release (`model/README.md`).

## 5. HPCC

Short runs work in a terminal on the development node `dev-amd24-h200`, after `source hpcc/activate.sh`; a development node stops a process after two hours of CPU time. Longer runs are batch jobs, submitted from a development node (the gateway refuses `sbatch`):

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
sbatch hpcc/run.sb run.py clips --checkpoint release
sbatch --time=04:00:00 hpcc/run.sb finetune.py --run grapes_v1
```

When `squeue --me` no longer lists the fine-tuning job, the masks and a video count with the fine-tuned model:

```bash
sbatch hpcc/run.sb run.py clips
sbatch hpcc/run.sb run.py video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

Logs are `sam3-<jobid>.out` in this folder. [`../hpcc/run_all.sh`](../hpcc/README.md) submits the first three jobs together with every other model's jobs and the leaderboard; [`../hpcc/count_rows.sh`](../hpcc/README.md) submits the last one for every video and every pipeline.

## 6. If something goes wrong

| What you see | Do this |
|---|---|
| `401` / `GatedRepoError` on download | Access to `facebook/sam3` not approved yet, or `hf auth login` not done on this machine. |
| `start_session` fails on a folder of frames | `run.py clips` retries with a temporary MP4 by itself. |
| `No module named sam3.train`, `psutil`, `einops`, `pycocotools` or `pkg_resources` | `python -m pip install -r requirements.txt`: it holds the `[train]` extra and the packages `sam3` imports without declaring them. |
| `Killed` in the middle of a run on a development node | The process used up the two hours of CPU time it gets there. Start from `source hpcc/activate.sh`, which limits the CPU threads, or submit the run as a batch job (section 5). |
| `CUDA out of memory` | `run.py`: shorten `--seconds`. `finetune.py`: `--resolution 768`, or a bigger GPU. |
| `No detector weights matched` after training | The trained checkpoint is kept in `../data/runs/<pipeline>_<run>/checkpoints/`; open an issue with the first lines of `unmatched_tensors` from the `.json`. |
| Masks look right but the leaderboard shows 0 tracks | Check that `../data/dataset/clips.json` lists the clip as `val` and `done`. |

Licence: Meta's SAM License. The clips and videos belong to National Grape.
