# MSU FRUIT · SAM 3.1 pipeline

**SAM 3.1** (Meta, March 2026) finds and follows every object that matches a **text prompt**, so it needs no boxes and no training to try on our video: give it "grape cluster" and it returns one mask per cluster, tracked through the clip. This folder runs it on our data and writes the masks where the SAM 2.1 pipeline's leaderboard scores them, so SAM 3.1 is compared with pretrained SAM 2.1, fine-tuned SAM 2.1 and YOLO26 on the same held-out clips with the same rules.

```
data/dataset (our labelled clips) ──▶ run_sam3.py clips ──▶ data/predictions/sam3_text/ ──▶ ../sam2_pipeline/fruit.py evaluate
data/videos/<file>                ──▶ run_sam3.py video ──▶ data/results/sam3/ (overlay video + counts per frame)
```

| File | What it is |
|---|---|
| `run_sam3.py` | The pipeline: `clips` (masks for the labelled val clips, for scoring) and `video` (overlay video and counts for a stretch of one video). |
| `requirements.txt` | Python packages; the install is below. |
| `hpcc/setup_env.sh`, `hpcc/run.sb` | Environment and batch job on the MSU HPCC. |
| `model/` | The SAM 3.1 weights (gated download, see [model/README.md](model/README.md)). Git ignores the `.pt`. |

## 1. What you need

- **A computer with an NVIDIA GPU** (CUDA 12.6 or newer): Google Colab (T4 is enough to try), the MSU HPCC, or a lab machine. SAM 3.1 does not run on a CPU or on a Mac.
- **Python 3.12 or newer, PyTorch 2.7 or newer** (Meta's requirements).
- **Access to the weights.** They are gated on Hugging Face: request access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) (and [facebook/sam3](https://huggingface.co/facebook/sam3) for the original SAM 3) with your Hugging Face account, wait for the approval email, then log in once on the computer that will run it: `hf auth login` (`pip install -U huggingface_hub` gives you the `hf` command; on an older install it is `huggingface-cli login`). The weights download on the first run, into Hugging Face's cache; `model/README.md` says how to keep a copy in `model/` instead.
- **Labelled clips in `data/dataset/`** for the `clips` job, from the SAM 2.1 labeller (`../sam2_pipeline/README.md`, section 1 or 2). The `video` job needs only a video in `data/videos/`.

## 2. Install

In a fresh Python 3.12 environment, on the GPU machine:

```bash
cd Capstone-MSU_Fruit/sam3_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
hf auth login
```

`requirements.txt` installs Meta's `sam3` package from its GitHub repository (`facebookresearch/sam3`) plus OpenCV and NumPy. Use the CUDA build of PyTorch that matches the machine's driver (`cu126` is the newest build that still runs on the HPCC's V100s; `cu128` is fine on newer cards).

**Colab:** *Runtime ▸ Change runtime type ▸ T4 GPU*, then in a cell `!git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit/sam3_pipeline && pip install -q -r requirements.txt`, then `from huggingface_hub import login; login()` with your token. Copy your `data/dataset/` (from Drive, where the SAM 2.1 notebook keeps it) into the clone's `data/` before running.

**MSU HPCC:** `bash hpcc/setup_env.sh` once on a development node (`ssh dev-amd20-v100`), then `~/envs/fruit_sam3/bin/hf auth login` (the environment is not active in your shell after the script), then `sbatch hpcc/run.sb clips`. The scripts mirror the SAM 2.1 ones in `../sam2_pipeline/hpcc/`.

## 3. Run

Check the install first (loads the model and runs it on nothing):

```bash
python -c "import sam3, torch; print('sam3 ok, CUDA:', torch.cuda.is_available())"
```

**Score it on our held-out clips.** This writes one PNG per frame, pixel value = object number, exactly the format of our labels:

```bash
python run_sam3.py clips
cd ../sam2_pipeline && python fruit.py evaluate
```

The second command scores `data/predictions/sam3_text/` next to every other model, on the *automatic* leaderboard (F1 at IoU 0.5, precision, recall, count error, tracks found vs true). The lineup entry is already active in `../sam2_pipeline/configs/lineup.yaml` (`SAM 3.1 · text 'grape cluster'`, type `folder`); an entry whose folder does not exist is skipped with a note, so it counts as soon as the masks are there. On the HPCC, `hpcc/run_all.sh` at the repository root runs this job together with the SAM 2.1 and YOLO26 fine-tuning and the leaderboard.

**Run it on a video** (30 seconds, 6 frames per second):

```bash
python run_sam3.py video ../data/videos/row3.mp4 --start 60 --seconds 30
```

Output: `data/results/sam3/row3_sam3.mp4` (masks drawn on the frames) and `row3_sam3.csv` (clusters visible per frame, distinct ids seen so far). Ids seen is not a verified cluster count, and a cluster count is not a yield estimate.

**Options for both jobs:** `--data <folder>` (the project folder, default `../data`, or `$PROJECT_DIR` as the HPCC jobs set it), `--text "green grapes"` (one concept phrase per prompt; on the team's sample clip wording changed YOLOE's confidence a lot, so try a few), `--prob 0.4` (output threshold), `--max-objects 128` (SAM 3.1's object capacity; Meta's default 16 is too few for a vine row), `--version sam3` (the original SAM 3), `--checkpoint model/sam3.1_multiplex.pt` (local weights; the default is the newest `.pt` in `model/`, else the package's download).

## 4. How it fits with the other pipelines

| | SAM 2.1 (`../sam2_pipeline`) | SAM 3.1 (here) | YOLO26 (`../yolo26_pipeline`) |
|---|---|---|---|
| Input it needs | a box or click per cluster (prompted), or a detector's boxes | a text prompt | nothing (trained on WGISD) |
| Training | fine-tuned on our labelled clips | none (zero-shot) | trained on WGISD, can be fine-tuned on our clips |
| Leaderboard | prompted (J&F) and, behind a detector, automatic | automatic | automatic |
| Runs on | laptop CPU, Colab, HPCC | NVIDIA GPU only | laptop CPU (slow), GPU |

The SAM 2.1 pipeline can also call SAM 3.1 directly (`type: sam3` in its lineup) when the `sam3` package is installed in *its* environment; this folder keeps SAM 3.1 in its own environment instead, so a broken `sam3` install never breaks the SAM 2.1 pipeline.

## 5. If something goes wrong

| What you see | Do this |
|---|---|
| `401` / `GatedRepoError` when the weights download | Your Hugging Face account is not approved for `facebook/sam3.1` yet, or `hf auth login` was skipped on this machine. |
| `start_session` fails on a folder of frames | SAM 3 wants a video file: `run_sam3.py clips` retries with a temporary MP4 of the frames by itself. If it still fails, note the error here. |
| `TypeError: … offload_state_to_cpu` on `start_session` | An old revision of the `sam3` package (Meta issue #544). `pip install -U --force-reinstall --no-deps git+https://github.com/facebookresearch/sam3.git`. |
| `CUDA out of memory` | Lower `--max-objects`, shorten `--seconds`, or ask for a bigger GPU (`sbatch --gpus=a100:1`). |
| Masks look right but the leaderboard shows 0 tracks | The PNGs must be named exactly like the frames (`00000.png` for `00000.jpg`); the script does this, so check that `data/dataset/clips.json` lists the clip as `val` and `done`. |

Licences: SAM 3 code and weights are under Meta's SAM License (research and commercial use allowed, see the repository). The team's clips and videos belong to National Grape.
