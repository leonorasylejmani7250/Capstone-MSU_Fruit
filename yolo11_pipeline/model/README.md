# yolo11_pipeline/model — the YOLO11 model the team ships

`best.pt` is the default model of every script in this pipeline and the "yolo11 · trained on WGISD" row of the leaderboard in `sam2_pipeline`. It is 5.8 MB and committed in git.

## best.pt

| | |
|---|---|
| Model | YOLO11n-seg (trained with Ultralytics 8.3.70; runs under 8.4), from Ultralytics' pretrained `yolo11n-seg.pt` |
| Trained on | WGISD train split, 94 photos; validated on 16 |
| Command | `python train.py` (50 epochs, imgsz 512, batch 2, seed 42, patience 20); the run is in `../../data/runs/yolo11/wgisd/` |
| Trained | October 2026, 4 GB NVIDIA GPU, 12 minutes |
| Validation (16 photos) | mask mAP@0.50 0.833 |
| **Test (27 held-out photos)** | **box mAP@0.50 0.810**, mAP@0.50:0.95 0.486 · **mask mAP@0.50 0.819**, mAP@0.50:0.95 0.469 |
| Count test (27 photos, conf 0.25) | MAE 2.59 clusters per photo; 451 predicted vs 407 by hand (best: conf 0.30, MAE 1.93) |

On WGISD photos it is scored at `imgsz` 512, the size it was trained at; on video the size depends on how big the clusters look in the footage, and `video.py` picks it (`--imgsz auto`). These are scores on WGISD photos; the number that matters is the leaderboard on our own clips (`python fruit.py evaluate` in `sam2_pipeline/`). Its per-frame counts on the partner's DJI videos are in `../../data/results/yolo11/CodyKresta_1025/`.

## Replacing it

Train and score a run ([`../README.md`](../README.md), step 3). If it is better, copy it here and update the table above; with a run named `wgisd_s1024`:

```bash
cd ~/Capstone-MSU_Fruit/yolo11_pipeline
cp ../data/runs/yolo11/wgisd_s1024/weights/best.pt model/best.pt
```
