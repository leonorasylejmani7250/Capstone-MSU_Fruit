# sam3_pipeline/model — SAM 3 weights

`run.py` uses the newest `.pt` in this folder, else it lets Meta's package download the release. `.pt` files here are ignored by git (3 GB each).

## Meta's release

Gated: request access at [facebook/sam3](https://huggingface.co/facebook/sam3), wait for the approval, then once on the machine that runs the pipeline: [`../../hpcc/README.md`](../../hpcc/README.md), step 1, has the blocks for the HPCC; [`../README.md`](../README.md), "On another computer", for any other machine. The file is `sam3.pt`.

## Fine-tuned on our clips

`python finetune.py --run <run>` writes `sam3_<run>.pt` here with a `.json` recipe (what it started from, the dataset, the settings). To share one, attach it to a GitHub Release of this repository (*Releases ▸ Draft a new release*) and put the link in the table. GitHub takes files up to 2 GiB, so the model goes up in parts (`split -b 1900M`), together with a `.sha256` checksum, its `.json` recipe and Meta's licence (`SAM_LICENSE.txt`, which has to go with every shared copy).

| File | Started from | Trained on | Leaderboard F1 (release → fine-tuned) | Download |
|---|---|---|---|---|
| `sam3_grapes_v1.pt` | Meta's `sam3.pt` | 1 labelled clip (8 frames, 315 clusters), 20 epochs, 11 Nov 2026 | 43.0 → 62.3 | [grapes_v1](https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/tag/grapes_v1) |

### Using a shared model

Download the parts into this folder, join them and check the result:

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline/model
curl -L -O https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/download/grapes_v1/sam3_grapes_v1.pt.part-aa
curl -L -O https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/download/grapes_v1/sam3_grapes_v1.pt.part-ab
curl -L -O https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/download/grapes_v1/sam3_grapes_v1.pt.sha256
curl -L -O https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit/releases/download/grapes_v1/sam3_grapes_v1.json
cat sam3_grapes_v1.pt.part-* > sam3_grapes_v1.pt
sha256sum -c sam3_grapes_v1.pt.sha256
rm sam3_grapes_v1.pt.part-*
```

The check prints `sam3_grapes_v1.pt: OK`. `run.py` then uses this model, the newest `.pt` here.
