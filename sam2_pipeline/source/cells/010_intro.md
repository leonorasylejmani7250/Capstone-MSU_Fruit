# MSU FRUIT · Fine-tune SAM 2.1 on our vineyard video, and score any model on the same clips

This notebook is one pipeline with two jobs:

1. **Train**: fine-tune Meta's SAM 2.1 (the newest SAM 2 release) on grape clusters that we label in our own videos.
2. **Evaluate**: grade *any* model on the same held-out clips with the same rules. SAM 2.1 before and after fine-tuning, YOLO / YOLOE, SAM 3, or anything else that can write a mask.

```
videos ──▶ short clips ──▶ labels ──▶ score the starting models ──▶ fine-tune SAM 2.1 ──▶ score again ──▶ run on new video
 Step 1      Step 2        Step 3            Step 4                      Step 5             Step 6          Step 7
```

## How to use it

**First run: the demo.** Make sure the runtime has a GPU (**Runtime ▸ Change runtime type ▸ T4 GPU**), leave `DEMO` ticked in cell A and choose **Runtime ▸ Run all**. The notebook downloads eight public grape photos with hand-drawn cluster masks, turns them into short panning clips, scores the pretrained models, fine-tunes for a few steps and scores again. Nothing of ours is touched. If the demo reaches Step 7, the setup works.

**Real run: our videos.**
1. In cell A untick `DEMO`, keep `SAVE_TO_DRIVE` ticked, then **Runtime ▸ Run all**. Approve the Google Drive prompt.
2. **Step 1** shows the `videos` folder. Put the videos there (upload button, drag and drop, or straight into Drive), then re-run Step 1 and Step 2.
3. **Step 3** is the only manual work: draw one box per grape cluster in each clip, check the result, press *mark done*.
4. From **Step 3c** on, choose **Runtime ▸ Run after** to score, train, score again and produce the overlay video.

Everything lands in one project folder: `videos/`, `dataset/` (frames and labels), `predictions/` (what each model drew), `models/` (fine-tuned weights), `results/` (leaderboards and videos).

**On the MSU HPCC** the same notebook runs in OnDemand Jupyter (set `PROJECT_DIR` in cell A), and long jobs go to the queue with `sbatch hpcc/run.sb train …` / `evaluate` / `video …`. Both use the same project folder layout and the same code. See Appendix E and `sam2_pipeline/README.md`.

## Three things to know before trusting a number

- **SAM 2 does not find grapes. It follows what it is shown.** Give it a box or a click on a cluster and it draws the mask and tracks it. So the leaderboard has two tables: *prompted* models (shown each cluster once) and *automatic* models (given only pixels). The automatic SAM 2 pipeline here is "a detector finds the clusters, SAM 2.1 segments and tracks them".
- **Scores are only as honest as the val clips.** Val clips must come from video the model never trained on, and every cluster in them must be labelled. Labels drafted by a model favour that model, which is why drafts stay unused until a person approves them.
- **Fine-tuning is not guaranteed to help.** It can only learn from the places where you corrected pretrained SAM 2.1 while labelling. Step 6 puts pretrained and fine-tuned side by side on the same clips so the answer is measured, not assumed.

*SAM 2.1 is the newest model in the SAM 2 line (checkpoints Sept 2024, code last updated Dec 2024). SAM 3 / SAM 3.1 (Nov 2025 / Mar 2026) is a newer, much larger family that also takes text prompts such as "grape cluster". This notebook trains SAM 2.1 and can score SAM 3.1 next to it (Appendix B).*
