# yolo11_pipeline/model — the YOLO11 model the team ships

`best.pt` is the default model of every script here and the "yolo11 · trained on WGISD" row of the SAM 2.1 leaderboard. It is 5.8 MB and committed in git.

## best.pt

| | |
|---|---|
| Model | YOLO11n-seg (Ultralytics 8.3.70), from Ultralytics' `yolo11n-seg.pt` |
| Trained on | WGISD train split, 94 images; validated on 16 |
| Command | `python train.py --epochs 50 --imgsz 512 --batch 2` (seed 42, patience 20); the run is in `../../data/runs/yolo11/wgisd/` |
| Trained | October 2026, 4 GB NVIDIA GPU, 12 minutes |
| Validation (16 images) | mask mAP@0.50 0.833 |
| **Test (27 held-out images)** | **box mAP@0.50 0.810**, mAP@0.50:0.95 0.486 · **mask mAP@0.50 0.819**, mAP@0.50:0.95 0.469 |
| Count test (27 images, conf 0.25) | MAE 2.59 clusters per image; 451 predicted vs 407 by hand (best: conf 0.30, MAE 1.93) |

The YOLO26 model in `../../yolo26_pipeline/model/` scores box 0.826 / mask 0.822 on the same test split; `python evaluate.py --model model/best.pt ../yolo26_pipeline/model/best.pt` puts the two side by side. This model's per-frame counts on the partner's DJI videos are in `../../data/results/yolo11/CodyKresta_1025/`.

Both models run at `imgsz` 512, the size they were trained at. These are scores on WGISD photos; the number that matters is the leaderboard on our own clips (`python fruit.py evaluate` in `sam2_pipeline/`).

## Replacing it

From `yolo11_pipeline/`: train, evaluate the run (`python evaluate.py --model ../data/runs/yolo11/<run>/weights/best.pt`), and if it is better:

```bash
cp ../data/runs/yolo11/<run>/weights/best.pt model/best.pt
```

Then update the table above.
