# Install · SAM 2.1 pipeline

Install, test the install, get the data, run the demo. No GPU and no partner data needed. Every command runs from this folder, `sam2_pipeline/`.

No install: open [`demo.ipynb` in Google Colab](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/demo.ipynb), *Runtime ▸ Run all*, approve the warning that the notebook is not from Google.

## 1. What you need

- Git: <https://git-scm.com/downloads>
- Conda: Miniforge (<https://conda-forge.org/download/>), Anaconda or Miniconda
- 3 GB of disk for the environment (6 GB on Linux) and 2 GB for the repository; internet on the first run
- Linux, macOS on Apple Silicon, or Windows (use the *Miniforge Prompt*; if anything fails, use WSL or Colab). Intel Macs: use Colab.

## 2. Get the code

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git
cd Capstone-MSU_Fruit/sam2_pipeline
```

## 3. Install

```bash
conda env create --prefix ./envs --file environment.yml
conda activate ./envs
```

5–15 minutes, mostly downloads (PyTorch, OpenCV). While pip works, conda shows only a spinner; leave it running.

On macOS 11 or 12 use this instead (OpenCV has no ready-made build for them):

```bash
PIP_ONLY_BINARY=opencv-python-headless conda env create --prefix ./envs --file environment.yml
```

To remove everything: `conda deactivate` then `rm -rf ./envs ../data/demo ~/.fruit_sam2`. To add a package: `pip install <pkg>`, then add it to `environment.yml` by hand.

## 4. Test the install

```bash
python tests/test_install.py
```

Ends with `INSTALL OK` after about a minute. The first run downloads Meta's SAM 2 code and the SAM 2.1 tiny weights (160 MB) into `~/.fruit_sam2/`.

## 5. Get the data

Demo data: nothing to do; the demo uses photos from `../data/wgisd/` (in the repository).

Partner video: copy it from the team drive into `../data/videos/` (`../data/README.md`).

## 6. Run the demo

```bash
jupyter notebook
```

Open `demo.ipynb`, **Run ▸ Run All Cells**. 5–10 minutes on a laptop CPU: it turns four photos into labelled clips, scores pretrained SAM 2.1 on a held-out clip, fine-tunes SAM 2.1 for 20 steps, scores again, and shows the true outlines next to what each model drew.

Results appear under the cells (the score table, **J&F** 0–100, and the comparison picture) and on disk in `../data/demo/`: `results/leaderboard_val.csv`, `predictions/<model>/`, `models/sam2.1_tiny_demo_run.pt`, `runs/demo_run/`. Delete `../data/demo/` to run it again from scratch. To keep the plots, **File ▸ Save and Export Notebook As ▸ HTML**.

Second notebook: `count_test.ipynb` runs our YOLO26 → SAM 2.1 on 27 WGISD photos against hand counts (a few minutes; results in `../data/count_test/results/`).

## 7. The full pipeline on our videos

| Where | How |
|---|---|
| Google Colab (free GPU) | Open [the pipeline notebook in Colab](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb); videos go to Google Drive. |
| This computer | `cd notebooks && jupyter notebook`, open the pipeline notebook, untick `DEMO`, set `PROJECT_DIR = "../../data"`. Command line: `python fruit.py status`. |
| MSU HPCC | `bash hpcc/setup_env.sh` once, then `sbatch hpcc/run.sb train …` |

[README.md](README.md) walks through all three.

## 8. If something goes wrong

| Problem | Fix |
|---|---|
| `conda: command not found` | Windows: open the *Miniforge Prompt*. macOS/Linux: `~/miniforge3/bin/conda init "$(basename "$SHELL")"` once, then a new terminal. |
| `Installing pip dependencies` spins for more than 30 minutes on a Mac | macOS 11 or 12: Ctrl+C, `rm -rf ./envs`, use the `PIP_ONLY_BINARY=…` command from section 3. |
| `conda env create` fails on the pip step (`ReadTimeoutError`) | `conda activate ./envs && pip install --timeout 120 -r requirements.txt`, then `pip install torch torchvision`. |
| `test_install.py` cannot download | Needs github.com and dl.fbaipublicfiles.com on the first run. |
| `Start Jupyter from the sam2_pipeline folder` | Run `jupyter notebook` from `Capstone-MSU_Fruit/sam2_pipeline`. |
| Anything else on Windows | WSL (Ubuntu), or Colab. |
