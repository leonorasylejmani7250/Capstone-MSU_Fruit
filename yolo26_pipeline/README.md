# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

A **CMSE 495 — Experiential Learning in Data Science** project at **Michigan State University**.

This repository develops a computer-vision workflow for detecting and segmenting fruit in field imagery and video. The current implementation focuses on **grape-cluster instance segmentation** with **YOLO11n-seg**, using the [WGISD (Wine Grape Instance Segmentation Dataset)](https://github.com/thsant/wgisd) for training and evaluation. Video inference supports optional object tracking, visual overlays, and per-frame CSV diagnostics.

The broader capstone aims to investigate scalable yield estimation and transfer to other fruits (including blueberries, apples, and peaches) and capture devices. **Those broader goals are not yet validated by the current grape-only implementation.** This folder, `yolo11_pipeline/`, holds the whole YOLO11 pipeline; install it with [INSTALL.md](INSTALL.md) here. The repository also holds a separate SAM 2.1 pipeline (`../sam2_pipeline/`, `../INSTALL.md`, `../demo.ipynb`; see `../README.md`). It is not a dependency of the scripts described here.

## Features

- Convert WGISD instance masks into YOLO segmentation polygon labels.
- Preview training, validation, or test annotations before training.
- Fine-tune a pretrained YOLO11n-seg model on grape clusters.
- Evaluate bounding-box and segmentation performance on a held-out WGISD test split.
- Run segmentation on a video, displaying masks, boxes, and confidence scores.
- Optionally track detected clusters across frames using ByteTrack.
- Save annotated `.mp4` videos and frame-by-frame `.csv` statistics.
- Extract frames from DJI footage for manual annotation and later fine-tuning.

> **Important:** The model detects **grape clusters**, not individual grape berries. `visible_now` counts detections in a single frame. Tracker IDs are not verified unique clusters or a yield estimate; repeated views, missed detections, and ID switches can cause errors.

## Repository layout

```text
Capstone-MSU_Fruit/yolo11_pipeline/
├── INSTALL.md                # Install and demo guide (macOS, Linux, Windows)
├── prepare_wgisd.py          # Convert WGISD instance masks into YOLO labels
├── preview_labels.py         # Visual quality check for converted labels
├── train.py                  # Train YOLO11n-seg
├── evaluate.py               # Evaluate on the WGISD test split
├── video.py                  # Annotated video inference and optional tracking
├── extract_dji_frames.py    # Sample DJI frames for manual labeling
├── requirements.txt         # Python dependencies
├── grape.yaml                # Dataset configuration (paths relative to this folder)
├── external/wgisd/           # WGISD source data
├── data/wgisd_seg/           # Converted images and segmentation labels
├── runs/                     # Training checkpoints, plots, evaluation artifacts
├── outputs/                  # Annotated results, CSVs, label previews
├── tests/                    # Project smoke test(s)
└── videos/                   # Local source videos; NOT committed to GitHub

The SAM 2.1 pipeline is in `../sam2_pipeline/` with its own install guide at `../INSTALL.md` (separate environment).
```

**Repository storage:** `main` includes the WGISD dataset, the converted labels, model checkpoints and label previews (about 1.4 GB to download, 3 GB on disk with git history). The annotated result videos (9.5 GB in Git LFS) are kept on the `Moe` branch only, so `main` needs no Git LFS. The `videos/` and `.venv/` directories are intentionally excluded.

## 1. Clone and set up the environment

### Prerequisites

- Python **3.11** (the version used for the documented Windows run).
- Git. [Git LFS](https://git-lfs.com/) is only needed to fetch the result videos from the `Moe` branch.
- A CUDA-capable NVIDIA GPU is recommended for training; CPU execution is also supported, but slower.
- Windows PowerShell for the commands below. Adjust activation and path separators on macOS/Linux.

### Clone the repository

```powershell
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git
cd Capstone-MSU_Fruit\yolo11_pipeline
```

If you already cloned the repository for the SAM 2.1 install (`../INSTALL.md`), use that copy and `cd` into `yolo11_pipeline`. The YOLO11 environment below is separate from the conda environment in `./envs`.

### Create a virtual environment (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**NVIDIA CUDA installation (tested project configuration):**

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These downloads are large, PyTorch and OpenCV (`cv2`) especially, and can take a while on a slow connection. If one stops with `ReadTimeoutError`, run the same command again with `--timeout 120` added ([INSTALL.md](INSTALL.md#3-install-python-dependencies) has the sizes).

**CPU-only setup:** Install an appropriate CPU build of PyTorch for your platform, then install `requirements.txt`. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for platform-specific instructions.

Check whether PyTorch detects your GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The project uses Ultralytics `8.3.70` with the **YOLO11n-seg** checkpoint. The Ultralytics package version and the model generation are different things.

## 2. Prepare the WGISD dataset

The preprocessing script expects the WGISD data in `external/wgisd/`, including `train_masked.txt`, `test_masked.txt`, and the corresponding `.jpg` and `.npz` files under `external/wgisd/data/`.

The converted labels and `grape.yaml` are committed, and `grape.yaml` uses paths relative to this folder, so this step is **optional**. Run it to rebuild the labels from the source masks (the conversion takes a few minutes):

```powershell
.\.venv\Scripts\python.exe prepare_wgisd.py
```

This creates or refreshes:

```text
data/wgisd_seg/images/{train,val,test}/
data/wgisd_seg/labels/{train,val,test}/
grape.yaml
```

**What it does:** `prepare_wgisd.py` rebuilds the labels from the source masks and rewrites `grape.yaml`. A different OpenCV version may trace a few outlines slightly differently, so `git status` can show small changes under `data/wgisd_seg/labels` afterwards; `git checkout -- data/wgisd_seg` restores the committed labels. The conversion uses the WGISD masked train/test lists, then splits part of training data into validation data with seed 42 and a default validation fraction of 0.15.

In the documented run, this produced **94 training, 16 validation, and 27 test images** with usable segmentation masks. Results can differ if the underlying dataset or conversion settings change.

### Preview the polygon labels

```powershell
.\.venv\Scripts\python.exe preview_labels.py --split val --limit 8
Start-Process ".\outputs\label_preview"
```

The preview images show the annotation polygons over grape clusters. Check these before running a long training job.

## 3. Train YOLO11n-seg

A short one-epoch smoke test:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 1 --imgsz 512 --batch 2
```

Full training configuration used in the project:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 50 --imgsz 512 --batch 2
```

Useful arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `50` | Maximum number of training epochs |
| `--imgsz` | `512` | Training image size |
| `--batch` | `2` | Training batch size |
| `--model` | `yolo11n-seg.pt` | Pretrained segmentation checkpoint |
| `--device` | `auto` | `auto`, a CUDA device (e.g., `0`), or `cpu` |

Training output is written beneath `runs/yolo11_wgisd/`; if that name already exists, Ultralytics generates a numbered directory such as `runs/yolo11_wgisd2/`. Use the corresponding `weights/best.pt` checkpoint for inference.

The 50-epoch run in this repository used:

```text
runs/yolo11_wgisd2/weights/best.pt
```

On a 4 GB NVIDIA GPU, keep the batch size small. If you get a CUDA out-of-memory error, lower `--batch` or `--imgsz`.

## 4. Evaluate the trained model

The evaluation script uses the **held-out WGISD test split**, not DJI video footage:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model "runs\yolo11_wgisd2\weights\best.pt" --imgsz 512 --batch 2
```

It prints bounding-box and segmentation **mAP@0.50** and **mAP@0.50:0.95**. Additional Ultralytics evaluation artifacts are saved under `runs/`.

The documented 50-epoch run reached approximately **83.3% validation mask mAP@0.50** on 16 WGISD validation images. That is **not** a test-set score or proof of accuracy on new DJI videos; run `evaluate.py` for test-set metrics.

## 5. Run inference on a video

Place source footage in a local directory, for example:

```text
videos/CodyKresta_1025/input.mp4
```

The source `videos/` folder is excluded from GitHub and must be supplied separately.

### Run an entire video (detection + segmentation)

```powershell
.\.venv\Scripts\python.exe video.py `
    --video "videos\CodyKresta_1025\input.mp4" `
    --model "runs\yolo11_wgisd2\weights\best.pt" `
    --mode detect `
    --conf 0.30 `
    --imgsz 768 `
    --max-frames 0 `
    --output "outputs\input_detect.mp4"
```

**Use `--max-frames 0` to process the entire video.** Without it, `video.py` processes at most 300 frames by default.

### Track objects across frames

```powershell
.\.venv\Scripts\python.exe video.py `
    --video "videos\CodyKresta_1025\input.mp4" `
    --model "runs\yolo11_wgisd2\weights\best.pt" `
    --mode track `
    --tracker bytetrack.yaml `
    --conf 0.30 `
    --imgsz 768 `
    --max-frames 0 `
    --output "outputs\input_track.mp4"
```

Tracking is experimental. IDs can change or disappear, and `tracker_ids_seen` must not be interpreted as a validated unique-cluster count.

### Video arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--video` | Required | Input video path |
| `--model` | Required | Path to `.pt` model weights |
| `--output` | `outputs/grapes.mp4` | Annotated MP4 destination |
| `--mode` | `detect` | `detect` or `track` |
| `--conf` | `0.30` | Minimum confidence (greater than 0, at most 1) |
| `--imgsz` | `768` | Inference image size (at least 160) |
| `--max-frames` | `300` | Number of frames to process; `0` = full video |
| `--start-sec` | `0` | Starting timestamp in seconds |
| `--tracker` | `bytetrack.yaml` | Tracker configuration for track mode |

### Process all videos in a directory (PowerShell)

```powershell
$python = ".\.venv\Scripts\python.exe"
$model = "runs\yolo11_wgisd2\weights\best.pt"
$inputDir = "videos\CodyKresta_1025"
$outputDir = "outputs\CodyKresta_1025\detect"

New-Item -ItemType Directory -Force $outputDir | Out-Null

$videos = @(Get-ChildItem $inputDir -File |
    Where-Object { $_.Extension -in '.mp4', '.mov', '.avi', '.mkv' } |
    Sort-Object Name)

foreach ($video in $videos) {
    $output = Join-Path $outputDir "$($video.BaseName)_yolo11.mp4"
    Write-Host "Processing $($video.Name)"

    & $python video.py `
        --video $video.FullName `
        --model $model `
        --mode detect `
        --conf 0.30 `
        --imgsz 768 `
        --max-frames 0 `
        --output $output

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to process $($video.Name)"
    }
}
```

### Output files and CSV columns

For an output named `outputs/input_detect.mp4`, the script also creates `outputs/input_detect.csv`.

| CSV column | Meaning |
| --- | --- |
| `frame` | Processed frame number, starting at 1 |
| `visible_now` | Detected grape clusters in the current frame |
| `tracked_now` | Detections with an assigned tracking ID in the current frame |
| `tracker_ids_seen` | Number of distinct tracker ID values observed so far (not ground-truth unique clusters) |

The annotated video includes translucent instance masks, bounding boxes, confidence percentages, and a frame-level statistics panel.

## 6. Extract DJI frames for future labeling

Use this utility to sample images for a manually annotated domain-adaptation dataset:

```powershell
.\.venv\Scripts\python.exe extract_dji_frames.py `
    --video "videos\CodyKresta_1025\input.mp4" `
    --every-sec 5 `
    --max-images 60
```

Frames are written to `data/dji_to_label/<video_stem>/`. This script only extracts images: it does **not** automatically label them or retrain the model. Manually annotate grape **clusters**, and keep an independent evaluation set when fine-tuning on new footage.

## 7. Large result videos stored in Git LFS

These videos are kept on the `Moe` branch only (`git lfs install`, then `git clone --branch Moe …`); `main` does not carry them, so a normal clone never downloads them.

Some generated MP4 files exceeded the single-file limit of the project's Git LFS storage configuration and were split into binary chunks such as:

```text
outputs/CodyKresta_1025/detect/example.mp4.part001
outputs/CodyKresta_1025/detect/example.mp4.part002
outputs/CodyKresta_1025/detect/example.mp4.parts.txt
```

The `.parts.txt` file records the original name and byte length. **A `.part001` file is not necessarily playable on its own.** To reconstruct a split video, create a temporary file named `reassemble.py` with the following content, replacing the example path:

```python
from pathlib import Path
import shutil

output = Path("outputs/CodyKresta_1025/detect/example.mp4")
parts = sorted(output.parent.glob(output.name + ".part[0-9][0-9][0-9]"))
if not parts:
    raise FileNotFoundError("No video chunks found")

with output.open("wb") as destination:
    for part in parts:
        with part.open("rb") as source:
            shutil.copyfileobj(source, destination)
print(f"Reconstructed {output} from {len(parts)} parts")
```

Run `python reassemble.py`. Make sure there is enough disk space for the reconstructed MP4. This is a byte-for-byte concatenation, not video transcoding.

## 8. Troubleshooting

| Symptom | Suggested fix |
| --- | --- |
| `Missing grape.yaml` or incorrect dataset path | Run `prepare_wgisd.py` on your machine. |
| CUDA is unavailable | Check your PyTorch build, NVIDIA driver, and CUDA-compatible installation. CPU inference remains available. |
| CUDA out of memory | Reduce `--batch` when training or `--imgsz` during inference. |
| `Missing trained model` | Use the actual `weights/best.pt` path in your local `runs/` directory; numbered run directories may differ. |
| Only 300 frames processed | Specify `--max-frames 0`. |
| `tracked_now` / IDs remain zero | You may be in `--mode detect`; tracking requires `--mode track`, and tracks are not always assigned. |
| Missing MP4 after clone | The result videos are only on the `Moe` branch: clone it with Git LFS installed (`git lfs pull` fetches them), and reassemble `.partNNN` chunks if applicable. |
| Inference video contains false positives or missed clusters | Validate on footage from your camera; consider annotating real-video frames for fine-tuning. WGISD validation scores do not guarantee field-video accuracy. |

## Dataset and acknowledgments

- **WGISD:** [thsant/wgisd](https://github.com/thsant/wgisd). Review its original license and citation guidance before redistributing or publishing dataset-derived artifacts.
- **Ultralytics YOLO:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).
- **PyTorch:** [pytorch.org](https://pytorch.org/).

Repository license: see [`LICENSE.txt`](../LICENSE.txt). Dataset and third-party model licenses may differ from the repository's own license.

## Course

**CMSE 495 — Experiential Learning in Data Science**  
**Michigan State University**
