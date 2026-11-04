# yolo26_pipeline/model — the YOLO26 model the team ships

`best.pt` is the default model of every script here and the "YOLO26n-seg · ours" row of the SAM 2.1 leaderboard. It is 6.6 MB and committed in git.

## best.pt

| | |
|---|---|
| Model | YOLO26n-seg (Ultralytics 8.4.198), from Ultralytics' `yolo26n-seg.pt` |
| Trained on | WGISD train split, 94 images; validated on 16 |
| Command | `python train.py --epochs 50 --imgsz 512 --batch 2` (seed 42, patience 20) |
| Trained | 2 November 2026, 2-core CPU, 80 minutes; best epoch 49 |
| Validation (16 images) | box mAP@0.50 0.832 · mask mAP@0.50 0.827 |
| **Test (27 held-out images)** | **box mAP@0.50 0.826**, mAP@0.50:0.95 0.497 · **mask mAP@0.50 0.822**, mAP@0.50:0.95 0.481 |
| Count test (27 images, conf 0.25) | MAE 3.56 clusters per image; 465 predicted vs 407 by hand (best: conf 0.35, MAE 2.48) |
| With SAM 2.1 behind it (`count_test.ipynb`) | MAE 2.59; 375 predicted vs 407 |

Baseline, the October **YOLO11n-seg** run in `../../data/results/yolo11/` (Ultralytics 8.3.70, same data and settings, Windows GPU): test box mAP@0.50 0.810, mask 0.819; count test MAE 2.59, 451 predicted (best: conf 0.30, MAE 1.93). `python evaluate.py --model model/best.pt ../data/results/yolo11/best.pt` puts the two side by side.

Both models run at `imgsz` 512, the size they were trained at. These are scores on WGISD photos; the number that matters is the leaderboard on our own clips (`python fruit.py evaluate` in `sam2_pipeline/`).

## Replacing it

From `yolo26_pipeline/`: train, evaluate the run (`python evaluate.py --model ../data/runs/yolo26/<run>/weights/best.pt`), and if it is better:

```bash
cp ../data/runs/yolo26/<run>/weights/best.pt model/best.pt
```

Then update the table above.
