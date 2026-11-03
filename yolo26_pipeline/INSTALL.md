# Install instructions · YOLO26 pipeline

These steps install the YOLO26 pipeline, test the install, and run its demo: evaluate the model in `model/best.pt` on public grape photos and run it on a short public video. It needs no GPU and no data from our community partner. Every command runs from this folder, `yolo26_pipeline/`.

The repository's other pipelines install separately, each with its own environment: [`../sam2_pipeline/INSTALL.md`](../sam2_pipeline/INSTALL.md) and [`../sam3_pipeline/README.md`](../sam3_pipeline/README.md). [`../README.md`](../README.md) describes all three.

## 1. What you need

- **Git**: <https://git-scm.com/downloads>.
- **Python 3.10 or newer** (3.12 is what was tested). `python3 --version` (macOS, Linux) or `py --version` (Windows) shows what you have; <https://www.python.org/downloads/> has installers.
- About **1 GB of disk** for the environment (about 5 GB on Linux, whose PyTorch build includes GPU libraries), plus about 2 GB for the repository (it includes the public WGISD grape dataset and its git history).
- **Internet access** on the first run (PyTorch, the packages, and Ultralytics' font and base weights).
- **macOS, Linux or Windows**; an NVIDIA GPU is optional (training is much faster with one).

## 2. Get the code

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git
cd Capstone-MSU_Fruit/yolo26_pipeline
```

If you already cloned the repository for another pipeline, use that copy. Every command below runs from this `yolo26_pipeline` folder.

## 3. Install the software

Create a virtual environment in this folder, install PyTorch, then the pipeline's packages. The environment lives in `.venv` (listed in `.gitignore`).

**macOS and Linux** (Terminal):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

**Windows** (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.\.venv\Scripts\python.exe` in place of `python` in every command.

`python -m pip install torch torchvision` gives a CPU build on macOS and Windows and a build with GPU libraries on Linux (about 4 GB of downloads there). For a smaller CPU-only build on Linux or Windows, or for a GPU build on Windows, take the command from <https://pytorch.org/get-started/locally/> instead of that line. The downloads are large (PyTorch, then OpenCV); on a slow connection they take a while, and if pip stops with `ReadTimeoutError`, run the same command again with `--timeout 120` added.

## 4. Test the install

```bash
python -c "import torch, ultralytics, cv2; print('ok · torch', torch.__version__, '· ultralytics', ultralytics.__version__, '· CUDA', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

The first line prints the versions (and whether a GPU is available; `False` is fine). The second runs the two unit tests, which end with `OK`.

## 5. Get the data

**Nothing to do.** The public WGISD grape dataset is in `../data/wgisd/`, and its YOLO labels in `../data/wgisd_yolo/`, both in git. The model to run, `model/best.pt`, is in git too. The partner's vineyard video is not: `../data/README.md` says where team members get it.

To rebuild the labels from the dataset's masks (optional, a few minutes; a different OpenCV version may move a few outline vertices, which `git checkout -- ../data/wgisd_yolo` undoes):

```bash
python prepare_wgisd.py
python preview_labels.py --split val --limit 8
```

The preview writes eight images to `../data/results/yolo26/label_preview/`; the yellow outlines should sit on the grape clusters.

## 6. Run the demo

**Evaluate the model on the 27 held-out WGISD test images** (about a minute on a CPU):

```bash
python evaluate.py
```

It prints box and mask mAP@0.50 and mAP@0.50:0.95 (the numbers in [model/README.md](model/README.md)) and saves plots to `../data/runs/yolo26/eval_test/`. On the first run Ultralytics downloads its plotting font (1 MB).

**Count clusters on the same images, against hand counts** (about a minute):

```bash
python count_test.py
```

It prints one line per image and a summary (mean absolute count error), and writes `../data/results/yolo26/count_test.csv` and the annotated images to `../data/results/yolo26/count_test/`.

**Run it on a video.** This makes a five-frame video from one public WGISD photo and runs the model on it, which checks video input and output without any partner footage:

macOS and Linux:

```bash
python -c "from pathlib import Path; import cv2; p=next(Path('../data/wgisd/data').glob('*.jpg')); im=cv2.resize(cv2.imread(str(p)),(640,480)); Path('../data/videos').mkdir(exist_ok=True); w=cv2.VideoWriter('../data/videos/public_demo.mp4',cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480)); [w.write(im) for _ in range(5)]; w.release(); print('wrote ../data/videos/public_demo.mp4')"
python video.py --video ../data/videos/public_demo.mp4 --imgsz 512 --max-frames 5
```

Windows:

```powershell
python -c "from pathlib import Path; import cv2; p=next(Path('../data/wgisd/data').glob('*.jpg')); im=cv2.resize(cv2.imread(str(p)),(640,480)); Path('../data/videos').mkdir(exist_ok=True); w=cv2.VideoWriter('../data/videos/public_demo.mp4',cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480)); [w.write(im) for _ in range(5)]; w.release(); print('wrote ../data/videos/public_demo.mp4')"
python video.py --video ../data/videos/public_demo.mp4 --imgsz 512 --max-frames 5
```

**Expected result:** `../data/results/yolo26/public_demo_detect.mp4` holds five annotated frames (masks, boxes, confidences, a count panel) and `public_demo_detect.csv` a header plus five rows. For a real video, put it in `../data/videos/` and add `--max-frames 0` ([README.md](README.md), section 4).

## 7. Beyond the demo

[README.md](README.md) in this folder covers training (`train.py`), tracking (`video.py --mode track`), frames for labelling (`extract_frames.py`), and running every video in a folder. The same model is also scored on the team's own labelled clips, next to SAM 2.1 and SAM 3.1, by the leaderboard in `../sam2_pipeline`.

**Every video in a folder (PowerShell):**

```powershell
Get-ChildItem ..\data\videos -File | Where-Object { $_.Extension -in '.mp4', '.mov', '.avi', '.mkv' } | ForEach-Object { python video.py --video $_.FullName --max-frames 0 }
```

## 8. If something goes wrong

| Problem | Fix |
| --- | --- |
| `python3: command not found` / `py` not found | Install Python from python.org and open a new terminal. |
| `.venv\Scripts\Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `.\.venv\Scripts\python.exe` instead of `python`. |
| pip stops with `ReadTimeoutError` | Run the same command again with `--timeout 120`; finished downloads come from pip's cache. |
| `No module named cv2` or `ultralytics` | The environment is not active: run the `activate` line from section 3 again. |
| `Missing weights: …/model/best.pt` | The clone is incomplete (`git status`), or you are on a branch without it. |
| Ultralytics pauses on "Downloading Arial.ttf" | It is fetching its plotting font; on a slow connection give it a minute. |
| Out of memory | Lower `--batch` for training or `--imgsz` for video. |

## 9. What was tested

This layout, from a fresh clone, on a 2-core Linux CPU with Python 3.12.16, PyTorch 2.14.1, Ultralytics 8.4.198 and OpenCV 5.0: sections 4, 5 and 6 as written, and the 50-epoch training run that made `model/best.pt`. The October YOLO11 pipeline (now archived under `../data/results/yolo11/`) was installed and run on Windows and Linux. macOS and Windows have not been run with this layout yet.
