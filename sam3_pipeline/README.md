# SAM 3.1 pipeline

SAM 3.1 segments and tracks every object that matches a text prompt ("grape cluster"), with no boxes and no training. This folder runs it on our clips and videos and writes its masks where the SAM 2.1 leaderboard scores them next to SAM 2.1 and YOLO26.

```
data/dataset (our labelled clips) ──▶ run_sam3.py clips ──▶ data/predictions/sam3_text/ ──▶ sam2_pipeline: python fruit.py evaluate
data/videos/<file>                ──▶ run_sam3.py video ──▶ data/results/sam3/ (overlay video + counts per frame)
```

| File | What it is |
|---|---|
| `run_sam3.py` | `clips`: masks for the labelled val clips. `video`: overlay video and counts for a stretch of one video. |
| `requirements.txt` | Python packages. |
| `hpcc/setup_env.sh`, `hpcc/run.sb` | Environment and batch job on the MSU HPCC. |
| `model/` | The SAM 3.1 weights ([model/README.md](model/README.md)). |

## 1. What you need

- An NVIDIA GPU with CUDA 12.6 or newer (Google Colab, the HPCC, a lab machine). Not a CPU, not a Mac.
- Python 3.12 or newer, PyTorch 2.7 or newer.
- Access to the weights: request it at [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) on Hugging Face, wait for the approval email, then log in once on the machine that runs it: `hf auth login`.
- For `clips`: labelled clips in `data/dataset/` from the SAM 2.1 labeller. For `video`: a video in `data/videos/`.

## 2. Install

```bash
cd Capstone-MSU_Fruit/sam3_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
hf auth login
```

Use the CUDA build of PyTorch that matches the machine (`cu126` runs on the HPCC's V100s; `cu128` on newer cards).

Colab: *Runtime ▸ Change runtime type ▸ T4 GPU*, then `!git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git && cd Capstone-MSU_Fruit/sam3_pipeline && pip install -q -r requirements.txt`, then `from huggingface_hub import login; login()`. Copy `data/dataset/` from Drive into the clone's `data/`.

HPCC: `bash hpcc/setup_env.sh` once on a development node, then `~/envs/fruit_sam3/bin/hf auth login`, then `sbatch hpcc/run.sb clips`.

## 3. Run

```bash
python -c "import sam3, torch; print('sam3 ok, CUDA:', torch.cuda.is_available())"
```

Masks for the held-out clips, then the leaderboard:

```bash
python run_sam3.py clips
cd ../sam2_pipeline && python fruit.py evaluate
```

The `SAM 3.1 · text 'grape cluster'` entry in `../sam2_pipeline/configs/lineup.yaml` scores `data/predictions/sam3_text/` on the automatic leaderboard (F1 at IoU 0.5, precision, recall, count error, tracks found vs true).

A stretch of video:

```bash
python run_sam3.py video ../data/videos/row3.mp4 --start 60 --seconds 30
```

Writes `data/results/sam3/row3_sam3.mp4` and `row3_sam3.csv` (clusters visible per frame, distinct ids so far).

Options: `--text "green grapes"` (one phrase per prompt), `--prob 0.4` (output threshold), `--max-objects 128` (object capacity; 16 is too few for a vine row), `--version sam3` (the original SAM 3), `--checkpoint <file>` (default: the newest `.pt` in `model/`, else the package's download), `--data <folder>` (project folder; default `../data` or `$PROJECT_DIR`).

## 4. Fine-tuning

This pipeline runs SAM 3.1 as released (zero-shot); nothing here trains it. Meta's `sam3` package ships training code for fine-tuning on an image dataset: `pip install -e ".[train]"` in the `sam3` repository, then `python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` with the data in Roboflow 100-VL layout (`train/`, `valid/`, `test/` folders with COCO annotations). Our clips are not exported in that layout yet, so fine-tuning SAM 3 is a separate piece of work; the SAM 2.1 pipeline is where fine-tuning happens today.

## 5. How it fits with the other pipelines

| | SAM 2.1 | SAM 3.1 | YOLO26 / YOLO11 |
|---|---|---|---|
| Input | a box or click per cluster, or a detector's boxes | a text prompt | nothing |
| Training | fine-tuned on our clips | none | trained on WGISD, fine-tuned on our clips |
| Leaderboard | prompted (J&F); automatic behind a detector | automatic | automatic |
| Runs on | laptop, Colab, HPCC | NVIDIA GPU | laptop, GPU |

## 6. If something goes wrong

| What you see | Do this |
|---|---|
| `401` / `GatedRepoError` on download | Access to `facebook/sam3.1` not approved yet, or `hf auth login` not done on this machine. |
| `start_session` fails on a folder of frames | `run_sam3.py clips` retries with a temporary MP4 by itself. |
| `TypeError: … offload_state_to_cpu` | Old `sam3` revision: `pip install -U --force-reinstall --no-deps git+https://github.com/facebookresearch/sam3.git` |
| `CUDA out of memory` | Lower `--max-objects`, shorten `--seconds`, or a bigger GPU (`sbatch --gpus=a100:1`). |
| Masks look right but the leaderboard shows 0 tracks | Check that `data/dataset/clips.json` lists the clip as `val` and `done`. |

Licence: Meta's SAM License. The clips and videos belong to National Grape.
