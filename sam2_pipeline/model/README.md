# sam2_pipeline/model — the fine-tuned SAM 2.1 the team ships

`python fruit.py video` uses the newest `sam2.1_*.pt` here by default, and `python fruit.py evaluate` scores every `.pt` here automatically.

## Putting a model here

1. Label clips ([`../../LABELLING.md`](../../LABELLING.md)).
2. Fine-tune ([`../README.md`](../README.md), step 3, or Step 5 of the notebook in Colab). The weights land in `data/models/sam2.1_base_plus_grapes_v1.pt` with a `.json` recipe.
3. The leaderboard ([`../README.md`](../README.md), step 4). Ship it only if it beats pretrained SAM 2.1 on the held-out clips.
4. Copy the `.pt` and `.json` here and fill in the table below.

## Sharing it

A SAM 2.1 checkpoint is 150–320 MB, over GitHub's 100 MB limit, so `.pt` files here are ignored by git. Attach the file to a GitHub Release of this repository (*Releases ▸ Draft a new release*) and put the link in the table. To use a shared one, download it from the release page into this folder.

## Models

| File | Trained on | Steps · size | J&F on the val clips (pretrained → fine-tuned) | Download |
|---|---|---|---|---|
| — | — | — | — | — |
