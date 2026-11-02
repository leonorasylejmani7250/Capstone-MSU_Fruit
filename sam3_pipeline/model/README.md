# sam3_pipeline/model — the SAM 3.1 weights

SAM 3.1 is used as Meta released it (no training by us), so this folder holds Meta's checkpoint, not a model of our own. The weights are **gated**: request access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) on Hugging Face, wait for the approval, then log in once on the machine that runs the pipeline (`hf auth login`).

With that done, `run_sam3.py` lets the `sam3` package download the checkpoint into Hugging Face's cache on the first run. To keep a copy here instead (handy on the HPCC, where compute nodes may have no internet), download it on a node that has:

```bash
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir model
```

`run_sam3.py` then uses the newest `.pt` in this folder by default (`--checkpoint` overrides it). The file is a few GB and is ignored by git (`*.pt` in `.gitignore`); only this README is committed. The original SAM 3 checkpoint is at [facebook/sam3](https://huggingface.co/facebook/sam3), for `run_sam3.py --version sam3`.

Licence: Meta's SAM License (see the `facebookresearch/sam3` repository).
