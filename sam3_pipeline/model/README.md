# sam3_pipeline/model — the SAM 3.1 weights

Meta's checkpoint, used as released. The weights are gated: request access at [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), wait for the approval, then `hf auth login` once on the machine that runs the pipeline.

`run_sam3.py` lets the `sam3` package download the checkpoint on the first run. To keep a copy here (compute nodes on the HPCC may have no internet), download it on a node that has:

```bash
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir model
```

`run_sam3.py` then uses the newest `.pt` here by default. The file is ignored by git. The original SAM 3 checkpoint is at [facebook/sam3](https://huggingface.co/facebook/sam3), for `run_sam3.py --version sam3`.
