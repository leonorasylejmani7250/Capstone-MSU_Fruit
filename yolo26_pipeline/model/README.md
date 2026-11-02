# yolo26_pipeline/model — the YOLO26 model the team ships

`best.pt` is the model every script in this folder uses by default (`evaluate.py`, `count_test.py`, `video.py`), and the one the SAM 2.1 leaderboard scores as "YOLO26n-seg · ours" (`../../sam2_pipeline/configs/lineup.yaml`). It is small (7 MB), so it is committed in git.

## best.pt

| | |
|---|---|
| Model | YOLO26n-seg (Ultralytics 8.4.198), fine-tuned from Ultralytics' `yolo26n-seg.pt` |
| Trained on | WGISD train split, 94 images (`../../data/wgisd_yolo`), validated on 16 |
| Command | `python train.py --epochs 50 --imgsz 512 --batch 2` (seed 42, patience 20; run `data/runs/yolo26/wgisd`) |
| Trained | 2 November 2026, on a 2-core Linux CPU, CPU_TRAIN_TIME |
| Best epoch | BEST_EPOCH of 50 (by validation mAP) |
| Validation (16 images) | box mAP@0.50 VAL_BOX_MAP50 · mask mAP@0.50 VAL_MASK_MAP50 |
| **Test (27 held-out images)** | **box mAP@0.50 TEST_BOX_MAP50, mAP@0.50:0.95 TEST_BOX_MAP · mask mAP@0.50 TEST_MASK_MAP50, mAP@0.50:0.95 TEST_MASK_MAP** (`python evaluate.py`) |
| Count test (27 images, conf 0.25) | mean absolute error COUNT_MAE clusters per image; COUNT_PRED predicted vs 407 counted by hand (`python count_test.py`) |

For comparison, the October **YOLO11n-seg** run (Ultralytics 8.3.70, same data and settings, Windows GPU), kept in `../../data/results/yolo11/`: test box mAP@0.50 0.810, mask 0.819; count test MAE 2.59, 451 predicted vs 407.

These are scores on WGISD photos (five varieties, Brazilian vineyards, 2017–2018), not on our partner's video. The number that matters for the project is the leaderboard on our own labelled clips, from `python ../sam2_pipeline/fruit.py evaluate`.

## Replacing it

Train (`python train.py …`), evaluate the new run (`python evaluate.py --model ../data/runs/yolo26/<run>/weights/best.pt`), and if it is better, copy it here and update the table above:

```bash
cp ../data/runs/yolo26/<run>/weights/best.pt model/best.pt
```

Keep the old model's numbers in the table's history, or move the old file to `../../data/results/` if it is worth keeping as a baseline.
