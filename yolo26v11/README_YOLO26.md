# YOLO26 Grape Cluster Segmentation — Project Update

This is an **add-on** for the existing MSU CMSE 495 grape detection project. It retains the YOLO11 baseline and uses the same WGISD labels. **Do not delete or replace the original scripts and weights.**

## What changes?

- New segmentation backbone: `yolo26n-seg.pt` (not `yolo26n.pt`, which is detection-only).
- Separate training runs: `runs/yolo26_wgisd*/weights/best.pt`.
- Dedicated scripts for training, evaluation, video processing, and YOLO11-versus-YOLO26 comparison.
- Existing `prepare_wgisd.py`, `grape.yaml`, and `video.py` are reused.

## Install on Windows (PowerShell)

In `D:\VSC\Python\Hacode\Grape_detect_clean`:

```powershell
py -3.11 -m venv .venv26
.\.venv26\Scripts\python.exe -m pip install --upgrade pip
.\.venv26\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
.\.venv26\Scripts\python.exe -m pip install -r requirements-yolo26.txt
.\.venv26\Scripts\python.exe -c "import torch,ultralytics;print(torch.cuda.is_available(),ultralytics.__version__)"
```

If the existing CUDA-enabled PyTorch is incompatible with the installed Ultralytics version on your machine, consult the official PyTorch install selector and install a compatible CUDA build. Avoid upgrading the existing `.venv` used for YOLO11.

Add `.venv26/` to `.gitignore` (and do not force-add the environment).

## Prepare WGISD

```powershell
.\.venv26\Scripts\python.exe prepare_wgisd.py
.\.venv26\Scripts\python.exe preview_labels.py --split val --limit 8
```

**Important:** `grape.yaml` contains an absolute path generated on the original machine. Re-run `prepare_wgisd.py` after moving/cloning this project to avoid using a stale path. The dataset preparation step regenerates the WGISD splits with a fixed seed and a train/val/test separation.

## Smoke test (1 epoch), then train

```powershell
.\.venv26\Scripts\python.exe train_yolo26.py --epochs 1 --imgsz 512 --batch 2 --name yolo26_smoke
.\.venv26\Scripts\python.exe train_yolo26.py --epochs 50 --imgsz 512 --batch 2 --name yolo26_wgisd
```

Default settings target a 4 GB NVIDIA laptop GPU. If an out-of-memory error occurs, try `--batch 1` or `--imgsz 416`.

If run name already exists, Ultralytics may create `yolo26_wgisd2`, etc. **Use the `best.pt` actually produced by the latest training run.** The model is trained from the COCO-pretrained YOLO26 segmentation checkpoint, not from YOLO11 weights.

## Evaluate on validation

```powershell
.\.venv26\Scripts\python.exe evaluate_yolo26.py --model runs\yolo26_wgisd\weights\best.pt --split val
```

Use `--split test` only for final held-out evaluation after choosing model settings.

## Process one full DJI video

```powershell
.\.venv26\Scripts\python.exe video_yolo26.py `
  --video "videos\CodyKresta_1025\DJI_20000722004218_0265_D.MP4" `
  --model "runs\yolo26_wgisd\weights\best.pt" `
  --mode detect --conf 0.30 --imgsz 768 --max-frames 0 `
  --output "outputs\yolo26\DJI_0265_full.mp4"
```

The output includes an annotated MP4 and same-basename CSV with `frame`, `visible_now`, `tracked_now`, `tracker_ids_seen`.

Optional YOLO26 NMS-free experimental inference: append `--nms-free` in **detect** mode. For a comparable baseline, first use the default one-to-many head (`nms` enabled).

## Process every DJI video

```powershell
.\.venv26\Scripts\python.exe batch_yolo26.py `
  --video-dir "videos\CodyKresta_1025" `
  --model "runs\yolo26_wgisd\weights\best.pt" `
  --mode detect --conf 0.30 --imgsz 768 `
  --output-dir "outputs\CodyKresta_1025\yolo26"
```

This processes **all frames** (`--max-frames 0` internally). Existing completed MP4/CSV pairs are skipped; add `--overwrite` to regenerate them.

## Compare YOLO11 and YOLO26 on identical WGISD images

```powershell
.\.venv26\Scripts\python.exe compare_yolo11_yolo26.py `
  --yolo11 "runs\yolo11_wgisd2\weights\best.pt" `
  --yolo26 "runs\yolo26_wgisd\weights\best.pt" `
  --split val --imgsz 512 --batch 2
```

Results: `outputs/yolo26/comparison.csv`. Compare instance mask mAP50/mAP50-95, precision, recall, inference time, and qualitative DJI results. Use the same image size and split for both models; training-library and recipe differences should be noted in your capstone report.

## Limitations and next steps

- WGISD validation/test performance is **not equivalent** to accuracy on DJI footage.
- The existing `render()` function reports the number of detected *grape clusters visible in the current frame*, not the number of berries or the yield in kilograms.
- Tracker IDs are **not** a verified count of unique clusters. An ID can change when detections are lost or reappear.
- Real-world validation requires labels sampled from the DJI videos. Consider manually annotating independent clips, then fine-tuning and reporting metrics on a held-out DJI set.
- YOLO26 improvements reported on COCO do **not** establish that YOLO26 outperforms YOLO11 on this WGISD/DJI setup. That must be measured.

## Technical sources

- https://docs.ultralytics.com/models/yolo26/
- https://docs.ultralytics.com/tasks/segment/
- https://docs.ultralytics.com/quickstart/
