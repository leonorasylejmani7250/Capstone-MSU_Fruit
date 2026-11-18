# Results: how well each model finds grape clusters

**Short version:** SAM 3 fine-tuned on our labelled clips is the best model that works without a person. It scores F1 62.2, up from 43.6 for SAM 3 as Meta released it. Even so, it finds only about half the grape clusters in any one frame, and every automatic model counts too few on average.

The numbers come from [`leaderboard_val.csv`](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/download/grapes_v1/leaderboard_val.csv) on the [`grapes_v1` release](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/tag/grapes_v1) (training round `grapes_v1`, November 2026). This page is written by hand from that file, so a new training round needs a new release and an update here.

## The test

- **5 val clips held out from training:** 80 frames, 230 labelled grape clusters, about 26 per frame. They come from other videos than the clip the models were fine-tuned on. One of the five has no labelled clusters, so it only tests whether a model marks something there.
- Every model is scored on the same clips and labels.
- There are two kinds of model, scored on different scales. Don't compare a number in one table with a number in the other.
  - **Automatic:** the model finds the clusters by itself. Counting fruit needs this.
  - **Prompted:** a person marks each cluster on the first frame, with a box or one click, and the model follows it through the clip.

## Automatic models

| Model | Trained on | F1 | Precision % | Recall % | Clusters found (of 230) | Count error / frame | Count bias / frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3, fine-tuned | our clips | 62.2 | 76.2 | 52.6 | 199 | 8.7 | −8.1 |
| SAM 3.1, fine-tuned | our clips | 60.2 | 75.0 | 50.3 | 140 | 9.4 | −8.7 |
| Our YOLO → fine-tuned SAM 2.1 | our clips (both) | 50.3 | 53.3 | 47.5 | 211 | 13.9 | −2.9 |
| Our YOLO → pretrained SAM 2.1 | our clips (detector) | 50.1 | 53.8 | 46.8 | 202 | 13.9 | −3.4 |
| SAM 3, as released | nothing | 43.6 | 78.5 | 30.2 | 88 | 16.2 | −16.2 |
| WGISD YOLO26 → pretrained SAM 2.1 | WGISD (detector) | 39.9 | 74.0 | 27.4 | 84 | 16.7 | −16.6 |
| SAM 3.1, as released | nothing | 36.6 | 73.9 | 24.4 | 67 | 17.6 | −17.6 |
| Our YOLO alone | our clips | 28.3 | 71.0 | 17.7 | 208 | 20.8 | −19.7 |
| YOLO11, WGISD then our clips | WGISD, then our clips | 14.1 | 79.1 | 7.8 | 109 | 23.9 | −23.7 |
| YOLO26, WGISD then our clips | WGISD, then our clips | 14.1 | 72.2 | 7.8 | 125 | 23.7 | −23.4 |
| YOLO26, WGISD only | WGISD | 8.1 | 71.4 | 4.3 | 68 | 24.7 | −24.7 |
| YOLO11, WGISD only | WGISD | 6.2 | 62.2 | 3.3 | 62 | 25.1 | −24.9 |
| YOLOE text prompt → SAM 2.1 | nothing (detector) | 0.8 | 100.0 | 0.4 | 1 | 26.2 | −26.2 |
| YOLOE text prompt alone | nothing | 0.0 | — | 0.0 | 0 | 26.3 | −26.3 |

- **Our YOLO** is Ultralytics' `yolo26n-seg` trained on our labelled clips (run `yolo_grapes_v1`). The CSV calls it `my YOLO`.
- **A → B** means model A finds the clusters, then SAM 2.1 (B) outlines them and follows them through the clip.
- **SAM 3 / SAM 3.1** are asked for "grape cluster" by text. **YOLOE** is asked the same way.
- **WGISD** is a public grape-image dataset (Embrapa Wine Grape Instance Segmentation Dataset), not our footage.

What the columns mean:

- **F1** (0–100): one score that balances precision and recall. Higher is better.
- **Precision:** of the clusters the model marked, the share that match a labelled cluster.
- **Recall:** of the labelled clusters in each frame, the share the model found.
- **Clusters found:** how many of the 230 labelled clusters the model picked up at some point in the clip.
- **Count error / frame:** the average gap between the model's count and the labelled count in a frame, whichever way it's off.
- **Count bias / frame:** the same gap keeping its sign, so too many in one frame cancels too few in another. Negative means the model counts too few on average.

## Prompted models (SAM 2.1 given a person's box or click)

| Model | Prompt | J&F | J | F |
| --- | --- | --- | --- | --- |
| SAM 2.1 base_plus, pretrained | box | 83.0 | 76.3 | 89.7 |
| SAM 2.1 base_plus, fine-tuned on our clips | box | 80.5 | 73.3 | 87.7 |
| SAM 2.1 base_plus, fine-tuned on our clips | click | 68.3 | 59.5 | 77.1 |
| SAM 2.1 base_plus, pretrained | click | 61.2 | 51.6 | 70.9 |

**J&F** (0–100) is the average of **J**, how much each mask overlaps the labelled mask, and **F**, how closely its outline follows the labelled outline. Higher is better.

## What the results say

1. **Fine-tuning SAM 3 on our footage pays off.** One labelled clip (8 frames, about 40 minutes of training on one A100 GPU) took SAM 3 from F1 43.6 to 62.2, the best of any automatic model. Fine-tuned SAM 3.1 is close behind (60.2, up from 36.6 as released).
2. **Every automatic model counts too few clusters on average.** The labels average 26.3 clusters per frame, and fine-tuned SAM 3 comes in about 8 short (count bias −8.1). It finds about half the clusters in any one frame (recall 52.6%), but it picks up 199 of the 230 at some point in the clip.
3. **A small average bias can hide large errors.** Our YOLO → SAM 2.1 is only about 3 clusters short on average, but in a single frame its count is off by about 14: it counts too many in some frames and too few in others, and the two cancel out. Nearly half of what it marks isn't a labelled cluster (precision 53–54%).
4. **Putting SAM 2.1 behind a detector helps a lot.** Behind our YOLO, SAM 2.1 lifts recall from 17.7% to about 47% (F1 28.3 → 50.3). Behind the WGISD YOLO26, it lifts recall from 4.3% to 27.4% (F1 8.1 → 39.9). SAM 2.1 follows each cluster the detector found into frames where the detector missed it. Precision drops (71% → 53–54% with our YOLO), so some of what it follows isn't a cluster.
5. **Training on public data alone doesn't carry over to this footage.** YOLO trained only on WGISD finds 3–4% of the clusters per frame (F1 6.2–8.1). Fine-tuning that model on our clips (F1 14.1) still ends up worse than fine-tuning Ultralytics' general `yolo26n-seg` (28.3). YOLOE, asked for "grape cluster" by text, finds nothing.
6. **Given a box per cluster, pretrained SAM 2.1 does best.** It scores J&F 83.0, and fine-tuning lowered that slightly (80.5). Given one click per cluster, fine-tuning helped (61.2 → 68.3).

## Things to keep in mind

- **The test is small:** 5 clips and 230 clusters ([LABELLING.md](LABELLING.md) asks for at least 8 val clips). A gap of a few points between two rows isn't a real ranking. Bigger gaps, like the 12 points between fine-tuned SAM 3 and our YOLO → SAM 2.1, are more convincing, but more labelled clips are needed to confirm them.
- **The SAM 2.1 box rows read high.** 4 of the 5 val clips were labelled with SAM 2.1's help, so SAM 2.1 is partly being compared with its own outlines.
- **The YOLO models trained on our clips are slightly favoured.** Their training picked its best epoch by scoring on these same val clips.
- **These are counts per frame, not per vine row.** Whole-video counts are the other result of the workflow, `data/results/cluster_counts_by_video.csv` ([README.md](README.md), step 5).
- **Speed is left out.** The CSV's `sec / frame` for the SAM 3 and SAM 3.1 rows only times reading masks saved earlier. Running SAM 3 itself took about 0.8–1.9 s per frame with clusters in view, on an H200 GPU.

## Where the numbers come from

The scores are in `leaderboard_val.csv` on the [`grapes_v1` release](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/tag/grapes_v1). On the HPCC the same table is `data/results/leaderboard_val.csv`, and `leaderboard_val_details.csv` beside it has the counts per clip and frame.

- **Scoring the models again:** [`sam2_pipeline/README.md`, step 4](sam2_pipeline/README.md#4-the-leaderboard). It needs the val clips and their labels, which aren't in this repository ([`data/README.md`](data/README.md)).
- **Adding a SAM 3 or SAM 3.1 model to the table:** its guide ([`sam3_pipeline/README.md`](sam3_pipeline/README.md), [`sam3_1_pipeline/README.md`](sam3_1_pipeline/README.md)) first writes the model's masks for the val clips, then runs the same step.
- **Getting the fine-tuned SAM 3 model:** it's on the same release, in two parts. [`sam3_pipeline/model/README.md`](sam3_pipeline/model/README.md#using-a-shared-model) has the block that downloads, joins and checks them. The model is shared under Meta's SAM License, which is on the release as `SAM_LICENSE.txt`.
- When the val labels change, the leaderboard drops its earlier rows, so every row in the file is scored on the same clips and labels.

<details>
<summary>Row names in the CSV</summary>

Some models appear twice in the CSV: once under the notebook's name, once under the command line's. Their scores agree to within half a point, and this page lists each model once, using the command line's row where there is one.

| On this page | Name(s) in the CSV |
| --- | --- |
| SAM 3, fine-tuned | `SAM 3 · text 'grape cluster' → fine-tuned on our clips (grapes_v1)` |
| SAM 3.1, fine-tuned | `SAM 3.1 · text 'grape cluster' → fine-tuned on our clips (grapes_v1)` |
| Our YOLO → fine-tuned SAM 2.1 | `my YOLO → SAM 2.1 base_plus · fine-tuned (grapes_v1)` |
| Our YOLO → pretrained SAM 2.1 | `my YOLO → SAM 2.1 base_plus pretrained` |
| SAM 3, as released | `SAM 3 · text 'grape cluster' · zero-shot` |
| WGISD YOLO26 → pretrained SAM 2.1 | `yolo26 · trained on WGISD → SAM 2.1 base_plus pretrained` |
| SAM 3.1, as released | `SAM 3.1 · text 'grape cluster' · zero-shot` |
| Our YOLO alone | `yolo26n-seg · pretrained → fine-tuned on our clips (yolo_grapes_v1)`; also `my YOLO` (F1 28.4) |
| YOLO11, WGISD then our clips | `yolo11 · WGISD → fine-tuned on our clips (yolo11_grapes_v1)` |
| YOLO26, WGISD then our clips | `yolo26 · WGISD → fine-tuned on our clips (yolo26_grapes_v1)` |
| YOLO26, WGISD only | `yolo26 · trained on WGISD` |
| YOLO11, WGISD only | `yolo11 · trained on WGISD` |
| YOLOE text prompt → SAM 2.1 | `YOLOE-26m 'grape cluster' → SAM 2.1 base_plus pretrained`; also `yoloe-26m-seg 'grape cluster' → SAM 2.1 base_plus pretrained` and `… · fine-tuned (grapes_v1)` (all F1 0.8) |
| YOLOE text prompt alone | `YOLOE-26m 'grape cluster' · zero-shot`; also `yoloe-26m-seg 'grape cluster'` |
| SAM 2.1, pretrained (box / click) | `SAM 2.1 base_plus · pretrained · box` / `· click` |
| SAM 2.1, fine-tuned (box / click) | `SAM 2.1 base_plus · fine-tuned on our clips (grapes_v1) · box` / `· click`; also `SAM 2.1 base_plus · fine-tuned (grapes_v1) · box` / `· click` (J&F 80.1 / 68.6) |

Other columns in the CSV:

- `J&F (DAVIS)`: J&F computed the way the DAVIS video benchmark does it.
- `IoU on prompt frame`: mask overlap on the frame where the person gave the prompt.
- `mask IoU (matched)`: how well the masks of correctly found clusters overlap the labels. For the YOLOE → SAM 2.1 rows this comes from a single cluster.
- `objects / frame (truth)`: labelled clusters per frame (26.3).
- `sec / frame`: see "Speed is left out" above.
- `data`: a fingerprint of the val clips and labels. It's the same (`c32d8744`) for every row.

</details>
