# sam3_1_pipeline/model — SAM 3.1 weights

`run.py` uses the newest `.pt` in this folder, else it lets Meta's package download the release. `.pt` files here are ignored by git (3 GB each).

## Meta's release

Gated: request access at [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), wait for the approval, then once on the machine that runs the pipeline: [`../../hpcc/README.md`](../../hpcc/README.md), step 1, has the blocks for the HPCC; [`../README.md`](../README.md), "On another computer", for any other machine. The file is `sam3.1_multiplex.pt`.

## Fine-tuned on our clips

`python finetune.py --run <run>` writes `sam3_1_<run>.pt` here with a `.json` recipe (what it started from, the dataset, the settings). To share one, attach it to a GitHub Release of this repository (*Releases ▸ Draft a new release*) and put the link in the table; to use a shared one, download it into this folder.

| File | Started from | Trained on | Leaderboard F1 (release → fine-tuned) | Download |
|---|---|---|---|---|
| — | — | — | — | — |
