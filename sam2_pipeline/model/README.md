# sam2_pipeline/model — the fine-tuned SAM 2.1 the team ships

This folder holds the fine-tuned SAM 2.1 weights we use for results. `python fruit.py video …` uses the newest `sam2.1_*.pt` here by default, and `python fruit.py evaluate` scores every `.pt` here automatically (box and click rows), next to pretrained SAM 2.1, YOLO26 and the rest of `configs/lineup.yaml`.

## How a model gets here

1. Label clips in `data/` (notebook Step 3 in Colab or OnDemand; `README.md` section 1 or 2).
2. Fine-tune: `sbatch hpcc/run.sb train --run grapes_v1 --steps 3000` on the HPCC, or Step 5 of the notebook in Colab. The weights land in `data/models/sam2.1_base_plus_grapes_v1.pt`, with a `.json` recipe (steps, learning rate, which clips) next to them.
3. Score it against pretrained SAM 2.1 and the other models on the held-out clips: `python fruit.py evaluate`. Ship it only if it wins there, and look at the comparison video in `data/results/` before trusting a gap of a point or two.
4. Copy the `.pt` and its `.json` into this folder, and fill in the table below.

## These files are too big for git

A SAM 2.1 base_plus checkpoint is about 320 MB (tiny: 150 MB). GitHub refuses files over 100 MB, and the team's Git LFS allowance is already used up, so `.pt` files in this folder are ignored by git (`*.pt` in `.gitignore`); only this README is committed. Share a model as an asset of a **GitHub Release** of this repository (*Releases ▸ Draft a new release ▸ attach files*; up to 2 GB per file, no download limit), and put the link in the table. Anyone can then fetch it into this folder:

```bash
curl -L -o model/sam2.1_base_plus_grapes_v1.pt https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/download/<tag>/sam2.1_base_plus_grapes_v1.pt
```

## Models

| File | Trained on | Steps · size | J&F on the val clips (pretrained → fine-tuned) | Download |
|---|---|---|---|---|
| — | — | — | — | — |
