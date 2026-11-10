# Labelling guide

You draw one box around each grape cluster, once. SAM 2.1 turns the box into an outline and follows the cluster through the clip. You check every frame and fix what is wrong. Labelled clips are what every model is fine-tuned on and scored against.

## The rule

1. One cluster, one box. Two clusters that touch are two if you can tell them apart. A cluster cut in two by a leaf, a shoot or a wire is one.
2. Label every cluster on the row in front of the camera, also the small and the half-hidden ones. Leave out the rows behind it.
3. A cluster keeps its outline and its number through the whole clip. If an outline jumps to another cluster, fix it.

## Open the labelling screen

On the MSU HPCC, after `bash hpcc/setup.sh` ([hpcc/README.md](hpcc/README.md)) and with video in `data/videos/`:

1. <https://ondemand.hpcc.msu.edu> ▸ *Interactive Apps* ▸ *Jupyter*. Jupyter location: *Conda Environment using Miniforge3 module*, environment `~/envs/fruit_sam2`. *Advanced Options*: 1 GPU. 2 to 4 hours. *Launch*, then *Connect*.
2. Open `Capstone-MSU_Fruit/sam2_pipeline/notebooks/MSU_FRUIT_SAM2_pipeline.ipynb`.
3. In cell **A · Settings** untick `DEMO` and set `PROJECT_DIR` to `~/Capstone-MSU_Fruit/data`.
4. Click the cell **Step 3b · Label**, choose *Run ▸ Run All Above Selected Cell*, then run the cell itself. The cells above it cut the videos into clips (16 frames each) the first time.

On a laptop or in Colab the same notebook works: [sam2_pipeline/README.md](sam2_pipeline/README.md).

## Label a clip

| To | Do this |
|---|---|
| Pick a clip | The **clip** menu: ⬜ to do, ✏️ draft, ✅ done. A name at the end means a teammate is on that clip: take another. |
| Label a cluster | Drag a box around it, on any frame. |
| Move through the clip | **◀ ▶** or the frame slider. A cluster that comes into view later: box it on that frame. |
| Fix a wrong outline | *a click will:* **Select**, click the cluster, press **Re-box selected**, drag the right box on that frame. |
| Nudge an outline | *a click will:* **Add to selected** or **Remove from selected**, then click the spot. |
| Remove a cluster | Select it, press **Delete selected**. |
| Take back the last change | **Undo**. |
| Finish the clip | Every cluster right in every frame: press **Clip checked: mark done**. |

Every change is saved at once. Only clips marked done are used.

## How many

The `val` clips first (the menu says `val` or `train`): at least 8, from at least 2 videos. Every model is scored on them. Then as many `train` clips as you can.

```bash
cd ~/Capstone-MSU_Fruit/sam2_pipeline
source hpcc/activate.sh
python fruit.py status
```

shows how many clips are done.

## A head start

In **Step 3a** tick `DRAFT_LABELS` and run the cell: our YOLO26 boxes the clusters in every clip that has no labels yet. Then open each clip in Step 3b, add the clusters it missed, fix the wrong ones and mark it done.

## Then

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1
```

fine-tunes every model on the labelled clips and scores them all ([hpcc/README.md](hpcc/README.md)).
