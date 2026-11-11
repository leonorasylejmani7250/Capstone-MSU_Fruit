# yolo26_pipeline/model — the YOLO26 model the team ships

`best.pt` is the default model of every script in this pipeline and the "yolo26 · trained on WGISD" row of the leaderboard in `sam2_pipeline`. It is 6.6 MB and committed in git.

## best.pt

| | |
|---|---|
| Model | YOLO26n-seg (Ultralytics 8.4.198), from Ultralytics' pretrained `yolo26n-seg.pt` |
| Trained on | WGISD train split, 94 photos; validated on 16 |
| Command | `python train.py` (50 epochs, imgsz 512, batch 2, seed 42, patience 20) |
| Trained | 2 November 2026, 2-core CPU, 80 minutes; best epoch 49 |
| Validation (16 photos) | box mAP@0.50 0.832 · mask mAP@0.50 0.827 |
| **Test (27 held-out photos)** | **box mAP@0.50 0.826**, mAP@0.50:0.95 0.497 · **mask mAP@0.50 0.822**, mAP@0.50:0.95 0.481 |
| Count test (27 photos, conf 0.25) | MAE 3.56 clusters per photo; 465 predicted vs 407 by hand (best: conf 0.35, MAE 2.48) |
| With SAM 2.1 behind it (`../../sam2_pipeline/count_test.ipynb`) | MAE 2.59; 375 predicted vs 407 |

On WGISD photos it is scored at `imgsz` 512, the size it was trained at; on video the size depends on how big the clusters look in the footage, and `video.py` picks it (`--imgsz auto`). These are scores on WGISD photos; the number that matters is the leaderboard on our own clips (`python fruit.py evaluate` in `sam2_pipeline/`). Its per-frame counts on the partner's DJI videos are in `../../data/results/yolo26/CodyKresta_1025/`.

## Replacing it

Train and score a run ([`../README.md`](../README.md), step 3). If it is better, copy it here and update the table above; with a run named `wgisd_s1024`:

```bash
cd ~/Capstone-MSU_Fruit/yolo26_pipeline
cp ../data/runs/yolo26/wgisd_s1024/weights/best.pt model/best.pt
```
