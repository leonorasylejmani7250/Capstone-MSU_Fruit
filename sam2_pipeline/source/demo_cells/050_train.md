## 4. Fine-tune SAM 2.1 on the three training clips
This runs **Meta's own SAM 2 training code** with the official fine-tuning recipe, scaled down to one small device. The demo does only 20 steps, enough to show the machinery working (about 1–2 minutes on a CPU). Real runs use thousands of steps on a GPU (Colab, or a batch job on the MSU HPCC). The result is a normal SAM 2.1 checkpoint in `data/demo_project/models/`.

The loss plot jumps around: each step trains on a different random clip with a different kind of hint (a mask, a box or clicks), and some of those are much easier than others. Twenty steps show no trend. The scores in the next step are what to judge a model by.
