# SAM 3.1 pipeline

SAM 3.1 (Meta, March 2026) segments and tracks every object that matches a text prompt ("grape cluster"), with no boxes drawn by anyone. It is SAM 3 with Object Multiplex, a faster tracker for many objects at once; the original SAM 3 is [`../sam3_pipeline/`](../sam3_pipeline/README.md). This folder counts the clusters in a video with it, scores it on our labelled clips next to every other model, and fine-tunes it on those clips.

```
../data/videos/<file>            ──▶ run.py video ──▶ ../data/results/sam3_1/<file>_sam3_1.mp4 + .csv, the total in ../data/results/cluster_counts.csv
../data/dataset (labelled clips) ──▶ run.py clips ──▶ ../data/predictions/sam3_1_text/ ──▶ the leaderboard in ../sam2_pipeline
../data/dataset (labelled clips) ──▶ finetune.py ──▶ model/sam3_1_<run>.pt ──▶ run.py clips ──▶ ../data/predictions/sam3_1_<run>/ ──▶ the leaderboard
```

## On the MSU HPCC

Set up once: [`../hpcc/README.md`](../hpcc/README.md), step 1, which builds the environment and puts Meta's weights in `model/`. Every block below can be pasted as it is into a terminal on the development node: log in, then `ssh dev-amd24-h200`.

### 1. Count the clusters in a video

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
source hpcc/activate.sh
python run.py video ../data/videos/2026/Grape--DJIOsmo/DJI_20010319022815_0391_D.MP4
```

A few minutes for a 20-second video. It ends with `Clusters counted: …`, writes `../data/results/sam3_1/<video>_sam3_1.mp4` (the video with every cluster tinted) and `.csv` (`frame`, `visible_now`, `clusters_so_far`), and adds the total to `../data/results/cluster_counts.csv`, the file every pipeline adds to. A total is the number of clusters the model followed, not a hand count, and not a yield estimate.

Every video of a folder, as a batch job (videos that have a total are skipped):

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
sbatch --time=04:00:00 hpcc/run.sb run.py video ../data/videos/2026/Grape--DJIOsmo
```

### 2. Score the released model on our labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
source hpcc/activate.sh
python run.py clips --checkpoint release
```

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

The first block writes the model's masks for the labelled val clips to `../data/predictions/sam3_1_text/`. The second is the leaderboard of every model, a few minutes in the terminal: this model is the row `SAM 3.1 · text 'grape cluster' · zero-shot`.

### 3. Fine-tune on our labelled clips

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
sbatch --time=04:00:00 hpcc/run.sb finetune.py --run grapes_v1
```

One batch job, about 40 minutes for a few labelled clips. To see how far it is (the first number in brackets is the epoch, 0 to 19):

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
squeue --me
grep "Train Epoch" "$(ls -t sam3_1-*.out | head -1)" | tail -n 1
```

When `squeue --me` no longer lists the job, its log ends with `merged … detector tensors` and the model is `model/sam3_1_grapes_v1.pt`. Score it:

```bash
cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
source hpcc/activate.sh
python run.py clips
```

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash --login hpcc/run.sb evaluate
```

Its row is `SAM 3.1 · text 'grape cluster' → fine-tuned on our clips (grapes_v1)`. From now on `run.py` uses the fine-tuned model, the newest file in `model/`, so step 1 counts with it; `--checkpoint release` after the command goes back to Meta's weights. A run name is trained once: after labelling more clips, use a new name such as `grapes_v2`.

## Reference

### `run.py`

- **Weights**: the newest `.pt` in `model/` (Meta's release after the setup, a fine-tuned model after `finetune.py`). `--checkpoint release` forces Meta's weights, `--checkpoint <file>` a particular file.
- **Options**, before or after the command name: `--text "green grapes"` (one phrase per prompt), `--prob 0.4` (output threshold; default 0.5), `--max-objects 128` (how many objects it can follow at once; default 64), `--data <folder>` (project folder; default `../data` or `$PROJECT_DIR`).
- **`video`**: the video goes through the model in stretches of 10 seconds that share one frame; a cluster keeps its id from one stretch to the next when its masks on the shared frame overlap. The total is the number of ids seen in at least two frames. `--start 60 --seconds 30` runs a stretch for a quick look; only a whole video adds its total. In a folder, `--overwrite` counts again the videos that have a total. A video's overlay and `.csv` are replaced by a later run with another model or prompt; each model's total stays in `cluster_counts.csv`.
- **`clips`**: each run writes `../data/predictions/<name>/` with a `recipe.json`, and the leaderboard adds a row for it by itself.

### `finetune.py`

It exports the labelled clips as a COCO dataset (`../data/sam3_dataset/`: train clips → `train/`, val clips → `test/`), runs Meta's trainer on it starting from the released weights (log, curves and checkpoint in `../data/runs/sam3_1_grapes_v1/`), and merges the fine-tuned detector into the released video checkpoint: `model/sam3_1_grapes_v1.pt` with a `.json` recipe. The trainer fine-tunes the detector, the part that finds clusters from the text; the tracker that follows them through the video stays as released. A job that runs out of time continues from its last finished epoch when the same line is submitted again.

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

With the defaults that is 2,000 training steps or more, about a second each on an A100 or H200. The fine-tuned file is 3 GB and ignored by git; share it through a GitHub Release ([model/README.md](model/README.md)).

### Files

| File | What it does |
|---|---|
| `run.py` | `video`: counts the clusters in a video, or in every video of a folder. `clips`: masks for the labelled val clips. |
| `finetune.py` | Fine-tunes the model on our labelled clips with Meta's trainer and writes the result to `model/`. |
| `finetune.yaml` | Meta's fine-tuning recipe with our dataset and settings (filled in by `finetune.py`). |
| `meta_trainer.py` | Starts Meta's trainer for `finetune.py`, with the one correction it needs to train. |
| `pipeline.py` | The names every script uses: model version, weights, data folder, where predictions and results go. |
| `model/` | The weights: Meta's release and our fine-tuned checkpoints ([model/README.md](model/README.md)). |
| `hpcc/` | On the HPCC: `activate.sh` for a terminal, `run.sb` for a batch job, `setup_env.sh` builds the environment. |
| `requirements.txt` | Python packages (Meta's `sam3` package with its trainer). |

## On another computer

It needs an NVIDIA GPU with CUDA 12.6 or newer (not a CPU, not a Mac), Python 3.12 or newer, and access to the weights: request it at [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) on Hugging Face and wait for the approval email. On a Linux machine, not on the HPCC:

```bash
cd Capstone-MSU_Fruit/sam3_1_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
hf auth login
```

```bash
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir model
```

Use the CUDA build of PyTorch that matches the machine (`cu126`, or `cu128` on newer cards). Then the commands are the ones above without the `source hpcc/activate.sh` and `sbatch` parts: `python run.py video <video>`, `python run.py clips`, `python finetune.py --run grapes_v1`, and `python fruit.py evaluate` in `../sam2_pipeline`.

Colab: *Runtime ▸ Change runtime type ▸ T4 GPU*, then `!git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit/sam3_1_pipeline && pip install -q -r requirements.txt`, then `from huggingface_hub import login; login()`. Copy `data/dataset/` from Drive into the clone's `data/`.

## If something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded` | You are on a gateway node: `ssh dev-amd24-h200`, then paste the block again. |
| `can't open file …/run.py`, or `No module named 'sam3'` | Paste the whole block: it starts with the `cd` and `source hpcc/activate.sh` lines this folder needs. |
| `401` / `GatedRepoError` on download | Access to `facebook/sam3.1` not approved yet, or `hf auth login` not done on this machine. |
| `Killed` in the middle of a run in the terminal | The process used up the two hours of CPU time it gets on a development node. Submit the run as a batch job: `sbatch --time=04:00:00 hpcc/run.sb` and then the command without `python`. |
| `CUDA out of memory` | `run.py`: lower `--max-objects`, shorten `--seconds`. `finetune.py`: `--resolution 768`, or a bigger GPU. |
| `No module named sam3.train`, `psutil`, `einops`, `pycocotools` or `pkg_resources` | `python -m pip install -r requirements.txt`: it holds the `[train]` extra and the packages `sam3` imports without declaring them. |
| `No detector weights matched` after training | The trained checkpoint is kept in `../data/runs/sam3_1_<run>/checkpoints/`; open an issue with the first lines of `unmatched_tensors` from the `.json`. |
| Masks look right but the leaderboard shows 0 tracks | Check that `../data/dataset/clips.json` lists the clip as `val` and `done`. |

Licence: Meta's SAM License. The clips and videos belong to National Grape.
