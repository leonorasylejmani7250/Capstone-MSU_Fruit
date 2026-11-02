# Install instructions

These steps install the SAM 2.1 pipeline, test the install, get the data and run the demo (`demo.ipynb`). The demo fine-tunes SAM 2.1 on sample grape images and scores it. It needs no GPU and no data from our community partner.

The repository's other pipeline, YOLO11, installs separately, with its own Python environment: see [yolo11_pipeline/INSTALL.md](yolo11_pipeline/INSTALL.md). [README.md](README.md) describes both.

**No-install option:** open [`demo.ipynb` in Google Colab](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/demo.ipynb), choose *Runtime ▸ Run all*, and approve the warning that the notebook is not from Google. It fetches the code by itself. Use this option if the local install below gives trouble.

## 1. What you need

- **Git**: <https://git-scm.com/downloads>.
- **Conda**, via Miniforge (recommended, <https://conda-forge.org/download/>) or Anaconda / Miniconda.
- About **3 GB of free disk** for the environment, plus 200 MB of downloads on first use, plus about 3 GB for the repository itself (1.4 GB to download; it includes the YOLO11 dataset and checkpoints). Linux needs more (about 6 GB), because the Linux PyTorch build includes GPU libraries.
- **Internet access** on the first run.
- **Linux, macOS on Apple Silicon (M1–M4), or Windows**; no GPU needed.
  - **Intel Macs** cannot install a recent enough PyTorch; use the Colab link above.
  - **On Windows**, run the commands below in the *Miniforge Prompt* (or *Anaconda Prompt*). Meta recommends WSL (Ubuntu) for SAM 2 on Windows; if anything fails natively, use WSL or the Colab link.
  - **Tested so far:** these steps were tested on Linux and on macOS (Apple Silicon). Windows is expected to work but has not been run yet.

## 2. Get the code

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git
cd Capstone-MSU_Fruit
```

Every command below runs from this `Capstone-MSU_Fruit` folder.

## 3. Install the software

```bash
conda env create --prefix ./envs --file environment.yml
conda activate ./envs
```

`environment.yml` lists everything:
- from conda-forge: Python 3.12, Jupyter, git, ffmpeg;
- from pip: PyTorch and the packages Meta's SAM 2 code and our pipeline use.

It takes 5–15 minutes, most of it downloading. The environment lives in `./envs` inside the repository (listed in `.gitignore`, so it is never committed).

**The pip step can look stuck while it downloads.** The largest downloads are PyTorch (about 125 MB on a Mac or Windows; about 4 GB on Linux, where it brings its GPU libraries) and OpenCV, the `cv2` module (45–60 MB). While pip works, conda shows only a spinning `Installing pip dependencies: \ | / -` line, with no progress bar and no file names; pip's messages appear once it finishes. On a slow connection the spinner can turn for 20 minutes or more. Leave it running: stopping it leaves a half-built `./envs`.

**On macOS 11 or 12, use this command instead** (your version is under Apple menu ▸ About This Mac). OpenCV's ready-made Mac builds need macOS 13 or newer. On older macOS, pip compiles OpenCV from source instead, which can take half an hour or more and needs a C++ compiler. This command makes pip install the newest OpenCV that has a ready-made build for your Mac (4.10):

```bash
PIP_ONLY_BINARY=opencv-python-headless conda env create --prefix ./envs --file environment.yml
```

**Clean up** whenever you like:
```bash
conda deactivate
rm -rf ./envs data/demo_project ~/.fruit_sam2
```
On Windows (Miniforge Prompt), the second line is `rmdir /s /q envs data\demo_project %USERPROFILE%\.fruit_sam2`.

**Changing the environment:** install with `conda install <pkg>` or `pip install <pkg>`, then add the package to `environment.yml` by hand so the file stays minimal. Do not regenerate the file with `conda env export`: that drops the pip packages, or pins everything to your machine.

## 4. Test the install

```bash
python tests/test_install.py
```

It ends with `INSTALL OK` after about a minute. On the first run it downloads two things, both cached in `~/.fruit_sam2/`:
- Meta's SAM 2 code, pinned to the version this pipeline was tested with;
- the SAM 2.1 tiny weights (160 MB).

It then checks that SAM 2.1 can outline and follow an object, and that the scoring code works.

## 5. Get the data

**Demo data**: nothing to do. `demo.ipynb` downloads four photos with hand-drawn cluster outlines from the public Embrapa WGISD dataset (CC BY-NC 4.0) and writes everything it makes to `data/demo_project/`, relative to the repository. `.gitignore` keeps `data/demo_project/` and `data/msu_fruit_project/` out of git.

**Project data** (vineyard video from National Grape) is not in git. The files are too large, and they belong to the community partner. Team members get them from the team's shared drive. Put them here:

```
Capstone-MSU_Fruit/data/msu_fruit_project/videos/     ← video files (.mp4, .mov, …), any number
```

The full pipeline then builds the rest of `data/msu_fruit_project/` (clips, labels, results). See section 7.

## 6. Run the demo

```bash
jupyter notebook
```

Jupyter opens in your browser. Open `demo.ipynb` and choose **Run ▸ Run All Cells**. It takes 5–10 minutes on a laptop CPU, and does five things:
1. turns the sample photos into short labelled clips;
2. scores pretrained SAM 2.1 on a held-out clip;
3. fine-tunes SAM 2.1 for 20 steps with Meta's training code;
4. scores the fine-tuned model on the same clip;
5. shows the true outlines next to what each model drew.

### Where the results are

**In the notebook**, under each cell as it finishes:

| Step | What appears |
|---|---|
| 2 | A table of the four clips (three for training, one held out for scoring), and four frames of the held-out clip with its true labels, one colour per grape cluster. |
| 3 | The results table, with one row: pretrained SAM 2.1. **J&F** is the main score, from 0 to 100 (higher is better). |
| 4 | The training loss over the 20 steps. It jumps around; that is expected. |
| 5 | The results table again, with a second row for the fine-tuned model. On a CPU both rows land near 92–93: 20 training steps on three clips are not meant to beat pretrained SAM 2.1, and a point either way is noise. |
| 6 | The main picture: the true outlines (top row), pretrained SAM 2.1 (middle row) and fine-tuned SAM 2.1 (bottom row) on four frames of the held-out clip. |

The plots are only in the notebook. To keep them, choose **File ▸ Save and Export Notebook As ▸ HTML**.

**On disk**, in `data/demo_project/` (Step 1 prints its full path):

| Path | What it holds |
|---|---|
| `results/leaderboard_val.csv` | The results table: one row per model, with J&F, J, F and seconds per frame. Steps 3 and 5 print this path. |
| `results/leaderboard_val_details.csv` | The same scores for each grape cluster. |
| `predictions/<model>/demo_CDY_2019_t0000000/` | What each model drew: one colour-coded mask per frame (`00000.png` to `00011.png`; black is background). |
| `models/sam2.1_tiny_demo_run.pt` | The fine-tuned SAM 2.1 tiny model (150 MB). |
| `runs/demo_run/` | The training run: its settings, its log (`console.log`), TensorBoard files and Meta's trainer checkpoints. |
| `dataset/`, `videos/` | The sample clips: frames (`dataset/JPEGImages/`), true labels (`dataset/Annotations/`) and short videos. |

**In Google Colab** the folder is `/content/Capstone-MSU_Fruit/data/demo_project/`. Click the folder icon in the left sidebar to browse it, and right-click a file ▸ *Download* to keep it: Colab deletes the folder when the session ends.

To run the demo again from scratch, delete `data/demo_project/`.

## 7. Beyond the demo: the full pipeline on our videos

| Where | How | When |
|---|---|---|
| **Google Colab** (free T4 GPU) | Open [`sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` in Colab](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb). Videos are uploaded to Google Drive. | Labelling and training. Nothing to install. |
| **This computer** | `cd sam2_pipeline/notebooks && jupyter notebook`, open the pipeline notebook, and in cell A untick `DEMO` and set `PROJECT_DIR = "../../data/msu_fruit_project"`. | Labelling and looking at results. Training is slow without a GPU. |
| **MSU HPCC** | From `sam2_pipeline/`: `bash hpcc/setup_env.sh` once on a dev node, then `sbatch hpcc/run.sb train …` | Long training runs on V100 / A100 / H200 GPUs. |

`sam2_pipeline/README.md` walks through all three. It also covers importing labels from Roboflow, CVAT or Label Studio, and adding any model to the scoring.

## 8. If something goes wrong

| Problem | Fix |
|---|---|
| `conda: command not found` | Windows: open the *Miniforge Prompt*. macOS / Linux: run `~/miniforge3/bin/conda init "$(basename "$SHELL")"` once (use `~/anaconda3` or `~/miniconda3` if that is where conda lives), then open a new terminal. |
| On a Mac, `Installing pip dependencies` spins for more than 30 minutes | Pip is probably compiling OpenCV from source, which happens on macOS 11 or 12. Press Ctrl+C, run `rm -rf ./envs`, then use the `PIP_ONLY_BINARY=…` command from section 3. |
| `conda env create` fails on the pip step (for example `ReadTimeoutError` on a slow connection) | Re-run the pip step yourself, where you can watch each download: `conda activate ./envs && pip install --timeout 120 -r sam2_pipeline/requirements.txt`, then `pip install torch torchvision`. Downloads that already finished come from pip's cache. |
| `test_install.py` cannot download | The first run needs internet access to github.com and dl.fbaipublicfiles.com (it falls back to a GitHub copy of the weights). |
| `Start Jupyter from the repository folder` | Jupyter was started elsewhere. Run `jupyter notebook` from `Capstone-MSU_Fruit`. |
| Anything else on Windows | Use WSL (Ubuntu), or the Colab link at the top. |
