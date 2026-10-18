# MSU FRUIT · SAM 2.1 pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moehocbaidi/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb)

Fine-tune **SAM 2.1** (the newest SAM 2 release) on our vineyard video, and score **any model** on the same held-out clips with the same rules. It runs in **Google Colab** and on the **MSU HPCC**, using the same code and the same project folder layout.

```
videos ──▶ short clips ──▶ labels ──▶ score the starting models ──▶ fine-tune SAM 2.1 ──▶ score again ──▶ run on new video
```

| File | What it is |
|---|---|
| `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` | The whole pipeline as a notebook with forms: upload videos, label, train, compare. Colab, or Jupyter on the HPCC. |
| `fruit_sam2.py` | The notebook's tool cells as one Python module (generated, do not edit by hand). |
| `source/` | Where the notebook and module come from: `cells/` (one file per notebook cell) and `build.py`. Edit a cell, run `python sam2_pipeline/source/build.py`, commit the regenerated files. |
| `fruit.py` | Command line around it, for batch jobs: `clips`, `import-coco`, `import-yolo`, `train`, `evaluate`, `video`, … |
| `hpcc/` | `setup_env.sh` (one-time setup), `run.sb` (any command as a GPU job), `config.sh` (your paths). |
| `configs/lineup.yaml` | Which models `evaluate` scores. Add any model here. |
| `examples/my_model.py` | Template for plugging in your own model. |

Everything a run produces lives in one **project folder**:

```
<project>/videos/                 your videos (never modified)
<project>/dataset/JPEGImages/     clips: <clip>/00000.jpg …
<project>/dataset/Annotations/    labels: <clip>/00000.png …  (pixel value = cluster number, 0 = background)
<project>/dataset/clips.json      split (train/val), source video and status (draft/done) of every clip
<project>/predictions/<model>/    what each model drew, same format as the labels
<project>/models/                 fine-tuned SAM 2.1 weights (+ a .json recipe for each)
<project>/results/                leaderboards (CSV), comparison videos, overlay videos and counts
<project>/runs/                   training logs and resumable checkpoints
```

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
git clone https://github.com/moehocbaidi/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit/sam2_pipeline
bash hpcc/setup_env.sh
```

The first line moves to a development node: these have internet access, compute nodes may not. `setup_env.sh` creates the conda environment and installs the SAM 2 code, the weights and a Jupyter kernel. To change `PROJECT_DIR` (default `~/msu_fruit_project`) or `SAM2_SIZE`, edit `hpcc/config.sh` before running it.

**Get videos onto the HPCC**, into `~/msu_fruit_project/videos/`, in one of these ways:
- **OnDemand**: *Files ▸ Home Directory ▸ msu_fruit_project ▸ videos ▸ Upload*. Best for files under about 1 GB.
- **Globus** (big files, or straight from Google Drive): the HPCC collection is `msuhpcc`. MSU's Google Drive collection is `msu#google`, so videos the team already has on Drive can go across without downloading them.
- **rsync** from your laptop: `rsync -av videos/ <netid>@rsync.hpcc.msu.edu:~/msu_fruit_project/videos/` (needs an SSH key).

**Use it from an HPCC shell**: `source hpcc/activate.sh` (from `sam2_pipeline/`) loads conda, activates the environment and reads `hpcc/config.sh`. Do that before typing any `python fruit.py …` command. Light commands are fine on a dev node; GPU work goes to the queue (below).

**Cut clips** (CPU only, fine on the dev node): `source hpcc/activate.sh && python fruit.py clips`

**Label** in OnDemand Jupyter, because labelling needs a screen:
1. Go to *Interactive Apps ▸ Jupyter*. Set *Jupyter Location* to *Conda Environment using Miniforge3 module* and the environment to `~/envs/fruit_sam2`.
2. Tick *Advanced Options*, set *Number of GPUs* to 1, and ask for 2–4 hours.
3. Open `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb`, using the kernel *Python (fruit_sam2)* if Jupyter asks.
4. In cell A, untick `DEMO` and set `PROJECT_DIR = "~/msu_fruit_project"`.
5. Run cell A, cell B and the Tools cells, skip Steps 1–2 (already done), and label in Step 3.

**Train, score and run as batch jobs**, submitted from the `sam2_pipeline` folder:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
sbatch hpcc/run.sb train --run grapes_v1 --steps 3000
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4 --start 60 --seconds 60 --weights models/sam2.1_base_plus_grapes_v1.pt
```

The first job fine-tunes SAM 2.1 and writes `models/sam2.1_base_plus_grapes_v1.pt`; the second writes the leaderboard to `results/leaderboard_val.csv`; the third runs the fine-tuned model on one minute of a video. `squeue --me` shows whether a job is running, and `tail -f fruit-<jobid>.out` follows its log.

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
The first command imports a COCO segmentation export, the second YOLO `.txt` labels (outlines or boxes), and the third accepts the SAM-drawn draft masks without checking them.
On the HPCC, run `source hpcc/activate.sh` first. Elsewhere, name the project before the command: `python fruit.py --project ../data/msu_fruit_project import-coco …`. In the notebook, the same tools are `import_coco(...)` and `import_yolo_seg(...)` (Appendix A).

- **Outlines** (polygons or masks) are used as they are.
- **Boxes only**, such as YOLO detection labels: SAM 2.1 draws each mask from its box, and the image stays a *draft*. Check drafts in the labelling screen, or approve them unchecked. Never feed rectangles in as masks, or SAM learns to draw rectangles.
- **Split by video, never by frame.** The importers read the source video off the file name (Roboflow's `clip_mp4-0012_jpg.rf.<hash>.jpg` works) and hold whole videos out for val. Roboflow's own train/valid folders mix frames of the same video, so they are ignored.
- Each image becomes a one-frame clip. Those train SAM 2.1 as an image model and count in the per-frame scores. The tracking score (J&F) needs real clips.

## 4. Scoring any model

Every model writes masks to `predictions/<model>/` in the label format, and one scoring function grades them all against the labelled **val** clips. There are two leaderboards:

- **Prompted** (SAM 2.1, pretrained or fine-tuned): the model is shown each cluster once, by a box or a click, and must follow it. Headline **J&F** (mask overlap + outline accuracy, 0–100), plus *IoU on prompt frame*.
- **Automatic** (YOLO, YOLOE, detector → SAM 2.1, SAM 3.1, anything else): the model gets only pixels. Headline **F1** at IoU 0.5, plus precision, recall, count error per frame, and tracks found vs true.

To add a model, add an entry under `models:` in `configs/lineup.yaml`, indented like the ones already there. Fine-tuned SAM 2.1 weights in `models/` are added automatically.

```yaml
models:
  - {name: my YOLO-seg, type: ultralytics, weights: runs/yolo_detector/weights/best.pt}
  - {name: YOLOE → SAM 2.1 fine-tuned, type: detect_then_sam2, checkpoint: models/sam2.1_base_plus_grapes_v1.pt,
     detector: {type: ultralytics, weights: yoloe-26m-seg.pt, text: grape cluster}}
  - {name: SAM 3.1 text, type: folder, path: predictions/sam3_text, tracks: true}     # PNGs written by another program
  - {name: my model, type: python, class: ../examples/my_model.py:MyModel}            # any Python class
```

**SAM 3.1** (March 2026, text prompts) needs Meta's `sam3` package and gated Hugging Face weights. Notebook Appendix B has a snippet that runs it in its own notebook and writes `predictions/sam3_text/`, which then scores like everything else. The built-in `type: sam3` adapter follows Meta's current API but was not run (no access to the gated weights).

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
- `demo.ipynb` at the repository root, run as a grader would.

**Not tested:** anything on a GPU, Colab itself, and the HPCC itself (`setup_env.sh`, `run.sb`, OnDemand). These scripts follow ICER's documentation as of October 2026. The SAM 3.1 adapter was not run either. Run the Colab demo once, and on the HPCC start with a 30-minute smoke test before a long job: `sbatch --time=00:30:00 hpcc/run.sb --size tiny train --run smoke --steps 20`.

## 7. When something goes wrong

| What you see | Do this |
|---|---|
| `setup_env.sh` fails at `pip install torch` | Run it on a dev node (they have internet). If its final check fails on a GPU node, ask ICER which CUDA version the drivers support. Then `pip uninstall -y torch torchvision`, change `cu126` in the script and re-run it. |
| `conda: command not found`, `ModuleNotFoundError`, "No project folder" | Run `source hpcc/activate.sh` first (from `sam2_pipeline/`). `ENV_DIR` in `hpcc/config.sh` must match where `setup_env.sh` built the environment. |
| A job ends at once with an empty log | Submit from `sam2_pipeline/` (`cd ~/Capstone-MSU_Fruit/sam2_pipeline`). |
| Job log: weights download fails | Compute nodes may be offline. Run `python fruit.py download --sizes <size>` on a dev node. |
| `CUDA out of memory` while scoring | Use smaller SAM 2.1 models (`size:` in `configs/lineup.yaml`), or give the job a bigger GPU (`--gpus=a100:1` or `h200:1`). Training handles memory by itself. |
| Labelling screen blank in Colab | Runtime ▸ Restart session, Run all. In OnDemand, use JupyterLab. |
| Fine-tuned model scores worse | Fewer steps, or more labelled clips; check the labels the comparison video shows going wrong. |

Licenses: SAM 2 code and weights are Apache 2.0. Ultralytics YOLO/YOLOE is AGPL-3.0 (fine for course and research use). The demo images are WGISD, CC BY-NC 4.0.
