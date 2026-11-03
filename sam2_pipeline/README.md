# MSU FRUIT · SAM 2.1 pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb)

Fine-tune **SAM 2.1** (the newest SAM 2 release) on our vineyard video, and score **any model** on the same held-out clips with the same rules. It runs on a laptop, in **Google Colab** and on the **MSU HPCC**, using the same code and the same project folder.

```
videos ──▶ short clips ──▶ labels ──▶ score the starting models ──▶ fine-tune SAM 2.1 ──▶ score again ──▶ run on new video
```

**New here? [INSTALL.md](INSTALL.md)** installs the pipeline and runs `demo.ipynb`, a 10-minute end-to-end demo on public sample grapes.

| File | What it is |
|---|---|
| `INSTALL.md`, `environment.yml` | Install guide and the conda environment (laptop). |
| `demo.ipynb` | Short end-to-end demo: score pretrained SAM 2.1, fine-tune it, score again, compare. |
| `tests/test_install.py` | One-minute check that the install works. |
| `count_test.ipynb` | Count test on the 27 WGISD test photos: detector (our YOLO26) → SAM 2.1 counts against hand counts, with overlays. The SAM 2.1 counterpart of `yolo26_pipeline/count_test.py`. |
| `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` | The whole pipeline as a notebook with forms: upload videos, label, train, compare. Colab, or Jupyter on the HPCC. |
| `fruit_sam2.py` | The notebook's tool cells as one Python module (generated, do not edit by hand). |
| `source/` | Where the notebooks and the module come from: `cells/`, `demo_cells/` and `count_test_cells/` (one file per notebook cell) and `build.py`. Edit a cell, run `python source/build.py`, commit the regenerated files. |
| `fruit.py` | Command line around it, for batch jobs: `status`, `clips`, `import-coco`, `import-yolo`, `train`, `evaluate`, `video`, … |
| `hpcc/` | `setup_env.sh` (one-time setup), `run.sb` (any command as a GPU job), `activate.sh`, `config.sh` (paths). |
| `configs/lineup.yaml` | Which models `evaluate` scores: pretrained SAM 2.1, our YOLO26, YOLOE, … Add any model here. |
| `model/` | The fine-tuned SAM 2.1 the team ships, with a README on how it was trained and how it scored. **Empty until the first real training run.** |
| `examples/my_model.py` | Template for plugging in your own model. |

**The project folder.** Everything a run reads and produces lives in one folder. In this repository that folder is [`../data/`](../data/README.md), shared with the YOLO26 and SAM 3.1 pipelines, and every `fruit.py` command uses it unless you pass `--project`:

```
data/videos/                 your videos (never modified)
data/dataset/JPEGImages/     clips: <clip>/00000.jpg …
data/dataset/Annotations/    labels: <clip>/00000.png …  (pixel value = cluster number, 0 = background)
data/dataset/clips.json      split (train/val), source video and status (draft/done) of every clip
data/predictions/<model>/    what each model drew, same format as the labels
data/models/                 fine-tuned SAM 2.1 weights from training runs (+ a .json recipe for each)
data/results/                leaderboards (CSV) and comparison videos; sam2/ holds the overlay videos and counts of `video` runs
data/runs/                   training logs and resumable checkpoints
```

In Colab the project folder is on Google Drive instead; on the HPCC it is the `data/` folder of the clone in your home directory. The demo keeps to `data/demo/`.

---

## 1. Colab (start here)

1. Click the *Open in Colab* badge at the top. Or upload `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` (*File ▸ Upload notebook* at colab.research.google.com), or double-click it in Google Drive. The notebook is self-contained; Colab needs nothing else from this repository.
2. **Runtime ▸ Change runtime type ▸ T4 GPU** (an L4 or A100 is faster if you have Colab Pro).
3. **Demo run first**: leave `DEMO` ticked in cell A and choose **Runtime ▸ Run all**. It downloads eight public grape photos with hand-drawn masks (Embrapa WGISD, CC BY-NC 4.0), turns them into short clips and runs every step. If it reaches Step 7, the setup works. This takes about 15–25 minutes on a T4.
4. **Real run**: untick `DEMO`, keep `SAVE_TO_DRIVE` ticked, **Run all**, and approve the Drive prompt. The project folder is *My Drive / msu_fruit_sam2*.
   - **Step 1 · upload videos**: tick `UPLOAD_NOW` and run the cell for files up to a few hundred MB. For big GoPro files, copy them into *My Drive / msu_fruit_sam2 / videos* from any computer (drive.google.com or Drive for Desktop), then re-run Step 1.
   - **Step 2** cuts clips (default 6 clips × 16 frames per video) and splits train/val. With 3+ videos, whole videos are held out.
   - **Step 3 · label**: drag a box around each grape cluster once; SAM 2.1 draws the mask and follows it through the clip. Fix wrong frames with *Re-box selected*, then *Clip checked: mark done*. Only clips marked done are used.
   - **Steps 4–7**, from Step 3c on: **Runtime ▸ Run after**. These steps score the pretrained models, fine-tune SAM 2.1, score again, and run on new video.

Colab disconnects after a while. Everything is on Drive, so **Run all** again picks up where you were, and an interrupted training run resumes.

## 2. MSU HPCC

You need an HPCC account. A faculty member (Dr. Bunting, or the course instructor) sponsors it at <https://contact.icer.msu.edu/account>.

**One-time setup** (10–20 min). Open a terminal on the HPCC: `ssh <netid>@hpcc.msu.edu`, or the shell in OnDemand (<https://ondemand.hpcc.msu.edu>, *Clusters* menu). Then:

```bash
ssh dev-amd20-v100
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash hpcc/setup_env.sh
```

The first line moves to a development node: these have internet access, compute nodes may not. `setup_env.sh` creates the conda environment and installs the SAM 2 code, the weights and a Jupyter kernel. The project folder is `~/Capstone-MSU_Fruit/data` (the clone's `data/` folder); to use another one, or another `SAM2_SIZE`, edit `hpcc/config.sh` before running it.

**Get videos onto the HPCC**, into `~/Capstone-MSU_Fruit/data/videos/`, in one of these ways:
- **OnDemand**: *Files ▸ Home Directory ▸ Capstone-MSU_Fruit ▸ data ▸ videos ▸ Upload*. Best for files under about 1 GB.
- **Globus** (big files, or straight from Google Drive): the HPCC collection is `msuhpcc`. MSU's Google Drive collection is `msu#google`, so videos the team already has on Drive can go across without downloading them.
- **rsync** from your laptop: `rsync -av data/videos/ <netid>@rsync.hpcc.msu.edu:~/Capstone-MSU_Fruit/data/videos/` (needs an SSH key).

**Use it from an HPCC shell**: `source hpcc/activate.sh` (from `sam2_pipeline/`) loads conda, activates the environment and reads `hpcc/config.sh`. Do that before typing any `python fruit.py …` command. Light commands are fine on a dev node; GPU work goes to the queue (below).

**Cut clips** (CPU only, fine on the dev node): `source hpcc/activate.sh && python fruit.py clips`

**Label** in OnDemand Jupyter, because labelling needs a screen:
1. Go to *Interactive Apps ▸ Jupyter*. Set *Jupyter Location* to *Conda Environment using Miniforge3 module* and the environment to `~/envs/fruit_sam2`.
2. Tick *Advanced Options*, set *Number of GPUs* to 1, and ask for 2–4 hours.
3. Open `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb`, using the kernel *Python (fruit_sam2)* if Jupyter asks.
4. In cell A, untick `DEMO` and set `PROJECT_DIR = "~/Capstone-MSU_Fruit/data"`.
5. Run cell A, cell B and the Tools cells, skip Steps 1–2 (already done), and label in Step 3.

**Train, score and run as batch jobs**, submitted from the `sam2_pipeline` folder:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v1 --steps 3000
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4 --start 60 --seconds 60 --weights models/sam2.1_base_plus_grapes_v1.pt
```

The first job fine-tunes SAM 2.1 and writes `data/models/sam2.1_base_plus_grapes_v1.pt`; the second writes the leaderboard to `data/results/leaderboard_val.csv`; the third runs the fine-tuned model on one minute of a video, with YOLOE finding the clusters (`--detector-weights-trained yolo26_pipeline/model/best.pt --detector-imgsz 512` uses our YOLO26 instead; a detector from `train-detector` is `runs/<name>/weights/best.pt`, trained at 1280). Its overlay video and counts land in `data/results/sam2/`. `sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt` fine-tunes our YOLO26 on the same labels. The repository's [`hpcc/run_all.sh`](../hpcc/README.md) submits all of these, plus SAM 3.1, as one chain. `squeue --me` shows whether a job is running, and `tail -f fruit-<jobid>.out` follows its log. When a fine-tuned model is the one the team wants to use, it goes to `model/` ([model/README.md](model/README.md) says how); `video` then uses it by default.

`run.sb` asks for one A100 for 4 hours. Change it per job, e.g. `sbatch --gpus=v100:1 --time=02:00:00 hpcc/run.sb evaluate`. The GPU types are v100/v100s (32 GB), a100 (40 or 80 GB), l40s (48 GB) and h200 (141 GB). Any of them works: training sizes itself to the card. It freezes the image backbone below 38 GB, trains in float16 on V100s and bfloat16 on the newer cards, and steps down frames per sample if memory runs out. Keep jobs at 4 hours or less so they can also use buy-in nodes. A run that hits the time limit resumes when you submit the same command again, on the same GPU type (a different type changes the training settings, so the run starts over).

Options that apply to every command go *before* the command name: `sbatch hpcc/run.sb --size large evaluate`, `python fruit.py --project ~/other_project status`.

`python fruit.py --help` lists every command. `python fruit.py status` shows what is in the project.

**Colab ↔ HPCC.** The project folder is plain files, so you can label in Colab (Drive), copy the folder to the HPCC with Globus, train there, and copy `models/` and `results/` back. Do not label the same clip in both places at once.

## 3. Labels you already have (Roboflow, CVAT, Label Studio, YOLO)

```bash
python fruit.py import-coco export/train/_annotations.coco.json
python fruit.py import-yolo export/train/images export/train/labels
python fruit.py approve --all-drafts
```
The first command imports a COCO segmentation export, the second YOLO `.txt` labels (outlines or boxes), and the third accepts the SAM-drawn draft masks without checking them. They go into the project folder, `../data`; on the HPCC, run `source hpcc/activate.sh` first. In the notebook, the same tools are `import_coco(...)` and `import_yolo_seg(...)` (Appendix A).

- **Outlines** (polygons or masks) are used as they are.
- **Boxes only**, such as YOLO detection labels: SAM 2.1 draws each mask from its box, and the image stays a *draft*. Check drafts in the labelling screen, or approve them unchecked. Never feed rectangles in as masks, or SAM learns to draw rectangles.
- **Split by video, never by frame.** The importers read the source video off the file name (Roboflow's `clip_mp4-0012_jpg.rf.<hash>.jpg` works) and hold whole videos out for val. Roboflow's own train/valid folders mix frames of the same video, so they are ignored.
- Each image becomes a one-frame clip. Those train SAM 2.1 as an image model and count in the per-frame scores. The tracking score (J&F) needs real clips.

## 4. Scoring any model

Every model writes masks to `data/predictions/<model>/` in the label format, and one scoring function grades them all against the labelled **val** clips. There are two leaderboards:

- **Prompted** (SAM 2.1, pretrained or fine-tuned): the model is shown each cluster once, by a box or a click, and must follow it. Headline **J&F** (mask overlap + outline accuracy, 0–100), plus *IoU on prompt frame*.
- **Automatic** (YOLO26, YOLOE, detector → SAM 2.1, SAM 3.1, anything else): the model gets only pixels. Headline **F1** at IoU 0.5, plus precision, recall, count error per frame, and tracks found vs true.

`python fruit.py evaluate` scores everything in `configs/lineup.yaml` plus every fine-tuned model automatically: SAM 2.1 weights in `model/` and in `data/models/`, and the YOLO-seg detectors that `train-detector` wrote to `data/runs/`. `python fruit.py train-detector --base ../yolo26_pipeline/model/best.pt` continues our WGISD-trained YOLO26 on the labelled clips, so the detector and SAM 2.1 both learn from the same labels. The lineup already lists our YOLO26 (`yolo26_pipeline/model/best.pt`); an entry whose file is not on this computer is skipped with a note. To add a model, add an entry under `models:`, indented like the ones already there. Paths are relative to the project folder, and a path that is not there is tried relative to the repository.

```yaml
models:
  - {name: my YOLO-seg, type: ultralytics, weights: runs/yolo_detector/weights/best.pt}
  - {name: YOLO26 → SAM 2.1 fine-tuned, type: detect_then_sam2, checkpoint: sam2_pipeline/model/sam2.1_base_plus_grapes_v1.pt,
     detector: {type: ultralytics, weights: yolo26_pipeline/model/best.pt, conf: 0.25}}
  - {name: SAM 3.1 text, type: folder, path: predictions/sam3_text, tracks: true}     # PNGs written by sam3_pipeline/run_sam3.py
  - {name: my model, type: python, class: ../examples/my_model.py:MyModel}            # any Python class
```

**SAM 3.1** (text prompts, gated weights, CUDA GPU) has its own folder, [`../sam3_pipeline/`](../sam3_pipeline/README.md). Its script writes `data/predictions/sam3_text/`, which the lineup entry above then scores like everything else. The built-in `type: sam3` adapter calls Meta's `sam3` package directly when it is installed in this environment; it follows Meta's current API but was not run (no access to the gated weights).

## 5. Reading the results honestly

- **Fine-tuning is not guaranteed to help.** It learns from the places where you corrected SAM 2.1 while labelling, which is why corrections matter. Compare fine-tuned and pretrained on the same val clips (`evaluate` does), and look at the comparison video before trusting a gap of a point or two.
- Labels drawn with SAM 2.1's help flatter SAM 2.1's *box* rows. Lean on the click rows and the automatic table, or hand-trace a few val frames without SAM assist for a neutral check.
- A cluster count is not yield. Converting counts to kilograms needs a calibration against hand-harvested vines.

## 6. What was tested

On a CPU-only Linux machine (Python 3.13, PyTorch 2.14, SAM 2 code at commit `2b90b9f`), using SAM 2.1 tiny at 512 px so it finishes on a CPU:
- the notebook end to end, in demo mode;
- every `fruit.py` command on a fresh project, including Meta's trainer on video and image clips together;
- the imports, where box → SAM 2.1 masks matched the true WGISD masks;
- metrics, which agree exactly with Meta's SA-V/DAVIS evaluator on 965 object-frames;
- the install test and Meta's trainer under the package versions Colab ships today (PyTorch 2.11, NumPy 2.1, ipywidgets 7);
- this layout (November 2026): `demo.ipynb` end to end with nbconvert (8.5 minutes on the CPU, J&F 92.7 pretrained / 92.6 fine-tuned); `count_test.ipynb` end to end (27 of 27 photos); `tests/test_install.py`; `fruit.py status` with the default project; `fruit.py evaluate` on the demo project with pretrained SAM 2.1, the fine-tuned demo weights, our YOLO26 and the YOLO11 baseline in one lineup (the YOLO rows need `imgsz: 512`, the size they were trained at); `fruit.py train-detector --base ../yolo26_pipeline/model/best.pt` for one epoch, and the automatic scoring of its result; the install test and `source/build.py` also from a fresh clone.

On macOS (Apple Silicon): the conda environment, the install test and `demo.ipynb` (October 2026, before the folders were reorganised).

**Not tested:** anything on a GPU, Colab itself, and the HPCC itself (`setup_env.sh`, `run.sb`, OnDemand). These scripts follow ICER's documentation as of October 2026. The SAM 3.1 adapter was not run either. Run the Colab demo once, and on the HPCC start with a 30-minute smoke test before a long job: `sbatch --time=00:30:00 hpcc/run.sb --size tiny train --run smoke --steps 20`.

## 7. When something goes wrong

| What you see | Do this |
|---|---|
| `setup_env.sh` fails at `pip install torch` | Run it on a dev node (they have internet). If its final check fails on a GPU node, ask ICER which CUDA version the drivers support. Then `pip uninstall -y torch torchvision`, change `cu126` in the script and re-run it. |
| `conda: command not found`, `ModuleNotFoundError` | Run `source hpcc/activate.sh` first (from `sam2_pipeline/`). `ENV_DIR` in `hpcc/config.sh` must match where `setup_env.sh` built the environment. |
| A job ends at once with an empty log | Submit from `sam2_pipeline/` (`cd ~/Capstone-MSU_Fruit/sam2_pipeline`). |
| Job log: weights download fails | Compute nodes may be offline. Run `python fruit.py download --sizes <size>` on a dev node. |
| `CUDA out of memory` while scoring | Use smaller SAM 2.1 models (`size:` in `configs/lineup.yaml`), or give the job a bigger GPU (`--gpus=a100:1` or `h200:1`). Training handles memory by itself. |
| Labelling screen blank in Colab | Runtime ▸ Restart session, Run all. In OnDemand, use JupyterLab. |
| Fine-tuned model scores worse | Fewer steps, or more labelled clips; check the labels the comparison video shows going wrong. |

Licenses: SAM 2 code and weights are Apache 2.0. Ultralytics YOLO/YOLOE is AGPL-3.0 (fine for course and research use). The demo images are WGISD, CC BY-NC 4.0.
