# SAM 2.1 pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonorasylejmani7250/Capstone-MSU_Fruit/blob/main/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb)

Label grape clusters in our video, fine-tune SAM 2.1 on the labels, score any model on the same held-out clips with the same rules, and count the clusters in a video. Runs on a laptop, in Google Colab and on the MSU HPCC.

```
videos ──▶ clips ──▶ labels ──▶ score the starting models ──▶ fine-tune SAM 2.1 ──▶ score again ──▶ count the clusters in new video
```

New here: [INSTALL.md](INSTALL.md) installs the pipeline and runs `demo.ipynb`.

| File | What it is |
|---|---|
| `INSTALL.md`, `environment.yml` | Install guide and conda environment (laptop). |
| `demo.ipynb` | 10-minute demo on public grapes: score pretrained SAM 2.1, fine-tune it, score again. |
| `count_test.ipynb` | Our YOLO26 → SAM 2.1 counts against hand counts on 27 WGISD photos. |
| `tests/test_install.py` | One-minute install check. |
| `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` | The whole pipeline as a notebook: upload videos, label, train, compare. Colab, or Jupyter on the HPCC. |
| `fruit.py` | The same pipeline as a command line: `status`, `clips`, `train`, `train-detector`, `evaluate`, `video`, `counts`, `check`, … (`python fruit.py --help`). |
| `fruit_sam2.py` | The notebook's tool cells as one module. Generated; do not edit. |
| `source/` | The notebooks' cells and `build.py`. Edit a cell, run `python source/build.py`, commit the regenerated files. |
| `configs/lineup.yaml` | The models `evaluate` scores. Add any model here. |
| `hpcc/` | HPCC setup (`setup_env.sh`), job script (`run.sb`), shell setup (`activate.sh`), paths (`config.sh`). |
| `model/` | The fine-tuned SAM 2.1 the team ships ([model/README.md](model/README.md)). |
| `examples/my_model.py` | Template for scoring your own model. |

**The project folder** is [`../data/`](../data/README.md) (every `fruit.py` command uses it unless you pass `--project`): `videos/`, `dataset/` (clips and labels), `predictions/`, `models/`, `results/`, `runs/`. In Colab it is on Google Drive; the demo keeps to `data/demo/`.

## Words used here

| Word | Meaning |
|---|---|
| **clip** | 16 frames cut from a video (Step 2), named after the video's path under `videos/`. Everything works on clips. |
| **label** | One mask per grape cluster per frame, drawn by SAM 2.1 from the box you draw (Step 3). Stored in `data/dataset/Annotations/`. |
| **draft / done** | A clip's labels are a *draft* until you check them and press *Clip checked: mark done* (or run `fruit.py approve`). Only *done* clips are used for training and scoring. |
| **train / val** | Whole videos are held out as *val*. Models are fine-tuned on the train clips and scored on the val clips. |
| **pretrained** | SAM 2.1 as Meta released it. |
| **trained on WGISD** | A YOLO model trained on the public WGISD photos (the two YOLO pipelines). |
| **fine-tuned on our clips** | A model trained further on our labelled clips: `fruit.py train` for SAM 2.1 (from pretrained), `fruit.py train-detector` for YOLO (from the WGISD model). |
| **zero-shot** | No training: SAM 3, SAM 3.1 or YOLOE finding clusters from the text "grape cluster". |
| **prompted / automatic** | The two leaderboards: a model shown each cluster once (SAM 2.1, J&F score) vs a model that must find them itself (everything else, F1 score). |

## 1. Colab

1. Click the badge above, or upload `notebooks/MSU_FRUIT_SAM2_pipeline.ipynb` at colab.research.google.com.
2. **Runtime ▸ Change runtime type ▸ T4 GPU**.
3. Demo run first: leave `DEMO` ticked in cell A, **Runtime ▸ Run all** (15–25 minutes on a T4).
4. Real run: untick `DEMO`, keep `SAVE_TO_DRIVE` ticked, **Run all**, approve the Drive prompt. The project folder is *My Drive / msu_fruit_sam2*.
   - **Step 1**: tick `UPLOAD_NOW` to upload videos, or copy big files into *My Drive / msu_fruit_sam2 / videos* and re-run the step.
   - **Step 2** cuts clips and splits train/val by video.
   - **Step 3**: drag a box around each cluster once; SAM 2.1 draws and follows the mask. Fix frames with *Re-box selected*, then *Clip checked: mark done*. Only clips marked done are used.
   - **Steps 4–7**: **Runtime ▸ Run after** from Step 3c: score, fine-tune, score again, run on new video.

Everything is on Drive, so after a disconnect **Run all** picks up where you were.

## 2. MSU HPCC

Set up once with `bash hpcc/setup.sh` in the repository folder; the whole workflow is in [`../hpcc/README.md`](../hpcc/README.md), labelling in [`../LABELLING.md`](../LABELLING.md).

**In every HPCC shell**, before any `fruit.py` command:

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py status
```

**GPU work goes to the queue**, submitted from `sam2_pipeline/`:

```bash
sbatch hpcc/run.sb train --run grapes_v1 --steps 3000
sbatch hpcc/run.sb train-detector --base ../yolo26_pipeline/model/best.pt --name yolo26_grapes_v1
sbatch hpcc/run.sb evaluate
sbatch hpcc/run.sb video videos/row3.mp4
```

In order: fine-tune SAM 2.1 from Meta's pretrained weights on our clips (→ `data/models/sam2.1_base_plus_grapes_v1.pt`); fine-tune the YOLO26 WGISD model on the same clips (→ `data/runs/yolo26_grapes_v1/weights/best.pt`); the leaderboard (→ `data/results/leaderboard_val.csv`); the cluster count of one video (section 5).

`squeue --me` shows jobs; `tail -f fruit-<jobid>.out` follows a log. `run.sb` asks for one A100 for 4 hours; override per job: `sbatch --gpus=v100:1 --time=02:00:00 hpcc/run.sb evaluate`. A training run that hits the time limit resumes when you submit the same command again. Options for every command go before the command name: `sbatch hpcc/run.sb --size large evaluate`. The project folder and the SAM 2.1 size are set in `hpcc/config.sh`.

## 3. Labels made elsewhere (Roboflow, CVAT, Label Studio, YOLO)

```bash
python fruit.py import-coco export/train/_annotations.coco.json
python fruit.py import-yolo export/train/images export/train/labels
python fruit.py approve --all-drafts
```

COCO or YOLO outlines are used as drawn. Box-only labels get a SAM 2.1 mask and stay drafts until checked in the labelling screen or approved with the third command. The split is by source video, read off the file names.

## 4. Scoring any model

`python fruit.py evaluate` scores every model in `configs/lineup.yaml` plus every model it finds in the project (SAM 2.1 weights in `model/` and `data/models/`, detectors from `train-detector` in `data/runs/`, SAM 3 and SAM 3.1 masks in `data/predictions/`), on the labelled val clips:

- **Prompted** (SAM 2.1): shown each cluster once by a box or a click, must follow it. Headline **J&F** (0–100).
- **Automatic** (YOLO26, YOLO11, YOLOE, detector → SAM 2.1, SAM 3, SAM 3.1): gets only pixels. Headline **F1** at IoU 0.5, plus precision, recall, count error, tracks found vs true.

Row names say what each model is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<run>)`. The table keeps the rows of earlier runs on the same clips, so models can be scored one at a time; `--fresh` starts it over (after renaming or removing an entry).

To add a model, add an entry under `models:` in `configs/lineup.yaml`. Paths are relative to the project folder, then to the repository; an entry whose file is missing is skipped.

```yaml
  - {name: my YOLO-seg, type: ultralytics, weights: runs/yolo_detector/weights/best.pt}
  - {name: YOLO26 → SAM 2.1 fine-tuned, type: detect_then_sam2, checkpoint: sam2_pipeline/model/sam2.1_base_plus_grapes_v1.pt,
     detector: {type: ultralytics, weights: yolo26_pipeline/model/best.pt, conf: 0.25}}
  - {name: masks from another program, type: folder, path: predictions/my_masks, tracks: true}
  - {name: my model, type: python, class: ../examples/my_model.py:MyModel}
```

## 5. Count the clusters in a video

```bash
python fruit.py video videos/row3.mp4
python fruit.py video videos/2026
python fruit.py counts
```

One video; every video in a folder; every model's total for every video. A detector finds the clusters and SAM 2.1 outlines and follows each one, so a cluster keeps one id for as long as it is in view; the total is the number of ids. Writes `data/results/sam2/overlay_<video>_0s.mp4`, `.counts.csv` (per frame) and `.tracks.csv` (one row per cluster), and adds the total to `data/results/cluster_counts.csv`, the file every pipeline adds to.

| Option | Default | Meaning |
|---|---|---|
| `--detector` | `yolo26_pipeline/model/best.pt` | The detector: any YOLO weights (`runs/<name>/weights/best.pt` for one fine-tuned on our clips), or `yoloe-26m-seg.pt`, which looks for `--text` |
| `--weights` | newest in `model/`, else pretrained | SAM 2.1 weights |
| `--conf` | `0.25` | Keep boxes that score at least this |
| `--detector-imgsz` | `auto` | The detector's input size (next section) |
| `--start`, `--seconds` | the whole video | A stretch of it, e.g. `--seconds 20` for a quick look; only a whole video adds its total to the table |

## 6. A model finds nothing

```bash
python fruit.py check videos/row3.mp4
```

It runs every detector on six frames of the video at every input size, prints a table (boxes per frame and top score), writes one picture per detector to `data/results/check/<video>/`, and says which detector and size to use, or that none works.

Three things make a detector silent:

| Cause | What `check` shows | Do this |
|---|---|---|
| The wrong weights: a base model such as `yolo26n-seg.pt`, which has no grape class, or a training run of an epoch or two | The weights line says so when the model loads | Use `yolo26_pipeline/model/best.pt` or a fine-tuned model |
| The wrong input size. A YOLO trained on photos only sees clusters about 40 to 100 px tall in the image it is given: footage shot closer needs a smaller size, wider shots a larger one | Boxes at some sizes, none at others | Nothing: the default size, `auto`, picks the size at which the detector is most sure |
| Footage unlike the photos the detector learned from, such as early-season video, where the clusters are flower clusters | No size has boxes, or a few low-scoring boxes sit on leaves | Label clips of this footage ([`../LABELLING.md`](../LABELLING.md)) and fine-tune |

## 7. Reading the results

- Fine-tuning is not guaranteed to help. Compare fine-tuned and pretrained on the same val clips, and watch the comparison video before trusting a gap of a point or two.
- Labels drawn with SAM 2.1's help flatter its *box* rows; lean on the click rows and the automatic table.
- A cluster total is the number of clusters a model followed. A cluster that is hidden and comes back is counted twice, a missed one not at all. The leaderboard's *tracks found / true* column is that check on the labelled clips; count one row by hand before relying on a video's total.
- A cluster count is not yield; converting counts to kilograms needs a calibration against hand-harvested vines.

## 8. When something goes wrong

| What you see | Do this |
|---|---|
| `setup_env.sh` fails at `pip install torch` | Run it on a dev node. If its final check fails on a GPU node, ask ICER which CUDA version the drivers support, `pip uninstall -y torch torchvision`, change `cu126` in the script, re-run. |
| `conda: command not found`, `ModuleNotFoundError` | `source hpcc/activate.sh` first, from `sam2_pipeline/`. |
| A job ends at once with an empty log | Submit from `sam2_pipeline/`. |
| Job log: weights download fails | Compute nodes may be offline: `python fruit.py download --sizes <size>` on a dev node. |
| `CUDA out of memory` while scoring | Smaller SAM 2.1 (`size:` in `configs/lineup.yaml`) or a bigger GPU (`--gpus=a100:1`). |
| Labelling screen blank in Colab | Runtime ▸ Restart session, Run all. In OnDemand, use JupyterLab. |
| Fine-tuned model scores worse | Fewer steps, or more labelled clips; check the labels the comparison video shows going wrong. |

Licences: SAM 2 code and weights Apache 2.0; Ultralytics YOLO/YOLOE AGPL-3.0; demo images WGISD, CC BY-NC 4.0.
