# yolo26_pipeline/model — the YOLO26 model the team ships

`best.pt` is the model every script in this folder uses by default (`evaluate.py`, `count_test.py`, `video.py`), and the one the SAM 2.1 leaderboard scores as "YOLO26n-seg · ours" (`../../sam2_pipeline/configs/lineup.yaml`). It is small (6.6 MB), so it is committed in git.

## best.pt

| | |
|---|---|
| Model | YOLO26n-seg (Ultralytics 8.4.198), fine-tuned from Ultralytics' `yolo26n-seg.pt` |
| Trained on | WGISD train split, 94 images (`../../data/wgisd_yolo`), validated on 16 |
| Command | `python train.py --epochs 50 --imgsz 512 --batch 2` (seed 42, patience 20; run `data/runs/yolo26/wgisd`) |
| Trained | 2 November 2026, on a 2-core Linux CPU, 80 minutes for the 50 epochs |
| Best epoch | 49 of 50 |
| Validation (16 images) | box mAP@0.50 0.832 · mask mAP@0.50 0.827 |
| **Test (27 held-out images)** | **box mAP@0.50 0.826**, mAP@0.50:0.95 0.497 · **mask mAP@0.50 0.822**, mAP@0.50:0.95 0.481 (`python evaluate.py`) |
| Count test (27 images, conf 0.25) | mean absolute error 3.56 clusters per image; 465 predicted vs 407 counted by hand (`python count_test.py`) |

For comparison, the October **YOLO11n-seg** run (Ultralytics 8.3.70, same data and settings, Windows GPU), kept in `../../data/results/yolo11/`: test box mAP@0.50 0.810, mask 0.819; count test MAE 2.59, 451 predicted vs 407 (`python count_test.py --model ../data/results/yolo11/best.pt`).

With SAM 2.1 behind it (`../../sam2_pipeline/count_test.ipynb`: YOLO26 boxes → pretrained SAM 2.1 tiny masks), the same 27 photos count MAE 2.59, 375 predicted vs 407: SAM 2.1 merges overlapping boxes, so it under-counts where YOLO26 alone over-counts.

So on WGISD the two are close: YOLO26 is a little better at finding and outlining clusters (mAP), YOLO11 a little better at counting at the default confidence. The count test depends strongly on `--conf`: YOLO26 is best at 0.35 (MAE 2.48, 384 predicted), YOLO11 at 0.30 (MAE 1.93, 415 predicted), so set the confidence per model before comparing counts. On the SAM 2.1 pipeline's demo clip (a held-out WGISD photo as a 12-frame pan) both score F1 ≈ 56 at IoU 0.5, YOLO26 finding 9 of 11 clusters as tracks and YOLO11 7; there, both must run at `imgsz: 512`, the size they were trained at, or they miss most clusters.

These are scores on WGISD photos (five varieties, Brazilian vineyards, 2017–2018), not on our partner's video. The number that matters for the project is the leaderboard on our own labelled clips, from `python fruit.py evaluate` in `sam2_pipeline/`.

## Replacing it

From `yolo26_pipeline/`: train (`python train.py …`), evaluate the new run (`python evaluate.py --model ../data/runs/yolo26/<run>/weights/best.pt`), and if it is better, copy it here and update the table above:

```bash
cp ../data/runs/yolo26/<run>/weights/best.pt model/best.pt
```

Keep the old model's numbers in this file, or move the old file to `../../data/results/` if it is worth keeping as a baseline.
