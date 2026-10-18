# Installation and Demo Guide

**Project:** Scalable Fruit Yield Estimation Using Video-Based Computer Vision (CMSE 495, Michigan State University)

This guide installs the **YOLO11n-seg grape-cluster segmentation** pipeline. The repository's other pipeline, SAM 2.1, installs separately with its own environment: see [INSTALL.md](INSTALL.md); [README.md](README.md) describes both. The model detects grape clusters, not individual berries, and its counts are not validated yield estimates.

> **Compatibility:** Python 3.11 is required for the setup below. CPU execution is the default demo route; a GPU is not required. macOS and Windows commands are provided, but those operating systems have not been tested in this verification environment.

## 1. Prerequisites

- Git
- Python **3.11**
- Internet access to install packages
- A terminal (macOS) or PowerShell (Windows)
- NVIDIA CUDA GPU optional; CPU inference is supported but slower

Run `git --version` and `python3.11 --version` (Mac) or `py -3.11 --version` (Windows) to verify installation. If missing, install Git and Python 3.11 first.

## 2. Clone the repository

In Terminal or PowerShell:

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git
cd Capstone-MSU_Fruit
```

The repository includes the WGISD dataset, the converted labels and the trained checkpoint (about 2 GB), so allow time for the download. It does not need Git LFS: the large result videos are kept on the `Moe` branch only. If you already cloned the repository for the SAM 2.1 install, use that copy. Run all remaining commands from `Capstone-MSU_Fruit`. Use a fresh virtual environment.

## 3. Install Python dependencies

### macOS (Terminal; CPU / Apple Silicon)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2
python -m pip install -r requirements.txt
python -m pip check
```

The macOS path has **not yet been verified** with the repository's pinned dependency versions. If PyTorch installation fails, consult the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the correct build for your machine, then rerun the requirements installation. Do not install the Windows-specific CUDA wheels on a Mac.

### Linux (Terminal; CPU demo)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

Use the macOS `python` commands in the remaining sections for Linux too.

### Windows (PowerShell; CPU demo)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Use the CPU setup above for the course demo. Install a different CUDA build only in a separate environment if GPU training is needed. Pinning PyTorch avoids changes in checkpoint-loading behavior in newer releases.

**Check dependencies:**

macOS (activated environment):

```bash
python -c "import torch, ultralytics, cv2; print('Dependencies imported successfully'); print('PyTorch:', torch.__version__)"
```

Windows:

```powershell
.\.venv\Scripts\python.exe -c "import torch, ultralytics, cv2; print('Dependencies imported successfully'); print('PyTorch:', torch.__version__)"
```

### Download the plotting font once

Ultralytics may pause while automatically fetching its plotting font. Download it explicitly before evaluation or training.

macOS / Linux:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; from ultralytics.utils import USER_CONFIG_DIR; p=Path(USER_CONFIG_DIR)/'Arial.ttf'; urlretrieve('https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf', str(p)); print(p)"
```

Windows:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from urllib.request import urlretrieve; from ultralytics.utils import USER_CONFIG_DIR; p=Path(USER_CONFIG_DIR)/'Arial.ttf'; urlretrieve('https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf', str(p)); print(p)"
```

## 4. Prepare the public grape dataset

The project uses the public [WGISD dataset](https://github.com/thsant/wgisd). The preparation script expects `external/wgisd/train_masked.txt`, `external/wgisd/test_masked.txt`, and matching image (`.jpg`) and mask (`.npz`) files under `external/wgisd/data/`.

The WGISD source data is included in the repository. If those files are missing after the clone, obtain the public dataset from the link above and place its files in `external/wgisd/` with the expected structure. **Do not use private partner data for the public demo.**

From the repository root:

macOS:

```bash
python prepare_wgisd.py
python preview_labels.py --split val --limit 8
```

Windows:

```powershell
.\.venv\Scripts\python.exe prepare_wgisd.py
.\.venv\Scripts\python.exe preview_labels.py --split val --limit 8
```

**This step is optional:** the converted dataset (`data/wgisd_seg/images/{train,val,test}/` with matching label folders) and `grape.yaml` are committed, and `grape.yaml` uses paths relative to the repository, so it works on any computer. Run `prepare_wgisd.py` only to rebuild the labels from the source masks; it rewrites `grape.yaml` the same way. A different OpenCV version may trace a few outlines slightly differently, so `git status` can show small changes under `data/wgisd_seg/labels` afterwards; `git checkout -- data/wgisd_seg` restores the committed labels. Dataset conversion can take several minutes and may print no progress until a split finishes. Label preview images should appear in `outputs/label_preview/`. Open those images to confirm polygons overlay grape clusters.

## 5. Optional: run a training installation test

Skip this section to run the existing trained model immediately in Section 6. A one-epoch training run tests the core dataset/model pipeline. It is **not** a full training run and may take many minutes on CPU. Full completion of this optional step was not verified here.

macOS:

```bash
python train.py --epochs 1 --imgsz 512 --batch 2 --device cpu
```

Windows:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 1 --imgsz 512 --batch 2 --device cpu
```

**Expected result:** The command completes without errors and writes a training run under `runs/`. If it fails, check the error and the troubleshooting section below. This test alone does not establish model accuracy.

## 6. Demo: visualize grape-cluster segmentation

The existing scripts support evaluation on public WGISD test images and annotated inference on video. For the course demo, use **public WGISD data** and an existing trained checkpoint; do not require private footage or a full 50-epoch training run.

The team's documented trained checkpoint is:

```text
runs/yolo11_wgisd2/weights/best.pt
```

Check that this file exists after cloning. If it does not, use a locally trained checkpoint from `runs/.../weights/best.pt`; a one-epoch checkpoint is suitable only for an installation smoke test, not for reporting final accuracy.

### Evaluate the existing model on the public test split

macOS:

```bash
python evaluate.py --model runs/yolo11_wgisd2/weights/best.pt --imgsz 512 --batch 2
```

Windows:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model "runs\yolo11_wgisd2\weights\best.pt" --imgsz 512 --batch 2
```

**Expected result:** The script prints bounding-box and mask mAP metrics and saves evaluation artifacts under `runs/`. The documented validation score is not a substitute for these held-out test metrics.

### Reproducible video installation test (public image)

Create a five-frame MP4 from one public WGISD image. This checks video input/output and model inference; it does not test tracking or performance on moving vineyard footage.

macOS (activated environment):

```bash
python -c "from pathlib import Path; import cv2; p=next(Path('external/wgisd/data').glob('*.jpg')); im=cv2.imread(str(p)); im=cv2.resize(im,(640,480)); Path('videos').mkdir(exist_ok=True); w=cv2.VideoWriter('videos/public_demo.mp4',cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480)); assert w.isOpened(); [w.write(im) for _ in range(5)]; w.release()"
python video.py --video videos/public_demo.mp4 --model runs/yolo11_wgisd2/weights/best.pt --mode detect --imgsz 512 --max-frames 5 --output outputs/public_demo_detect.mp4
```

Windows:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import cv2; p=next(Path('external/wgisd/data').glob('*.jpg')); im=cv2.imread(str(p)); im=cv2.resize(im,(640,480)); Path('videos').mkdir(exist_ok=True); w=cv2.VideoWriter('videos/public_demo.mp4',cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480)); assert w.isOpened(); [w.write(im) for _ in range(5)]; w.release()"
.\.venv\Scripts\python.exe video.py --video videos/public_demo.mp4 --model runs/yolo11_wgisd2/weights/best.pt --mode detect --imgsz 512 --max-frames 5 --output outputs/public_demo_detect.mp4
```

**Expected result:** `outputs/public_demo_detect.mp4` contains five annotated frames, and `outputs/public_demo_detect.csv` contains a header and five data rows. These visible-cluster counts are not unique-cluster totals or yield estimates. For a real video demonstration, substitute an authorized public or team-owned sample and increase `--max-frames`.

## 7. Troubleshooting

| Problem | Suggested fix |
| --- | --- |
| `python3.11` or `py -3.11` not found | Install Python 3.11 and reopen the terminal. |
| `grape.yaml` was edited to an absolute path | Run `prepare_wgisd.py` again from the repository root; it writes relative paths. |
| Missing `best.pt` | Check the actual run folder; update the model path. |
| Out of memory | Reduce `--batch` for training or `--imgsz` for inference. |
| CUDA not available | Use CPU or install the platform-compatible PyTorch build. |
| macOS dependency errors | Record the error and update/test a Mac-compatible dependency setup before submission. |

## 8. Verification notes

Checked against the public `Moe` branch at commit `b5a70a66d69bc3883cf09eea4105189de664b830` on October 17, 2026. The trained `runs/yolo11_wgisd2/weights/best.pt` checkpoint is included directly in Git. The dataset-only Git LFS pull succeeded.

On October 18, 2026 the YOLO11 files were merged into `main`, with the result videos left on the `Moe` branch and the one dataset file that was in Git LFS stored directly, so `main` needs no Git LFS. The evaluation command above was re-run from `main` on Linux (CPU) with the same test scores.

A fresh Linux CPU environment using Python 3.11.17, PyTorch 2.2.2, torchvision 0.17.2 and the repository requirements passed dependency compatibility checks, both existing smoke tests, full dataset preparation (94 training / 16 validation / 27 test images), eight label previews, evaluation on all 27 test images, and the five-frame public-image video test. The optional training command initialized and processed its first batch; a complete epoch was not verified. Windows and macOS still require a separate clean-machine test. Observed test mAP50 was 0.8103 for boxes and 0.8192 for masks. These are execution checks on the included checkpoint and split, not an independent audit of training provenance or yield accuracy.

## 9. Sharing and reproducibility

- Use **relative paths** within the repository; `grape.yaml` is committed with paths relative to the repository, so do not replace them with personal absolute paths.
- Do not commit confidential partner data, raw private videos, or other NDA/IP-protected assets.
- Review WGISD, pretrained model, and project license terms before redistribution.
- Keep `.venv/`, local footage, and other machine-specific files out of Git.
- Have at least one other teammate clone and follow this guide on a clean machine, ideally a different OS. Record any corrections in this file.
- Ensure the final README links to `INSTALL.md`.

For advanced options (full training, tracking, DJI frame extraction, and processing many videos), see the existing project `instructions.md` or README documentation.
