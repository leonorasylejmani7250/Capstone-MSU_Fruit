"""Template: score your own model with the pipeline.

List it in a lineup file and run `python fruit.py evaluate`:

    - name: my own model
      type: python
      class: ../examples/my_model.py:MyModel
      args: {threshold: 0.5}

or, in the notebook:  benchmark({"my own model": MyModel(threshold=0.5)})

The contract is one method. predict(clip, prompts) gets a clip and returns a [T, H, W] integer array:
0 = background, 1, 2, 3 … = one number per grape cluster. Keep a cluster's number the same from frame to frame
if your model tracks (set tracks = True); otherwise any numbering per frame is fine.

`clip.image(t)` is frame t as an RGB uint8 array [H, W, 3]; `len(clip)` is the number of frames; `clip.hw` is (H, W).
"""
import numpy as np

try:                                   # command line / HPCC: the pipeline module
    from fruit_sam2 import Model
except ImportError:                    # notebook: the pipeline put its Model class here before loading this file
    Model = globals().get("Model", object)


class MyModel(Model):
    needs_prompts = False   # True: gets one hint per object (like SAM 2) and is scored on the prompted table
    tracks = False          # True: the same cluster keeps the same number across frames
    prompt = None           # for prompted models: "box" | "point" | "mask"

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def prepare(self):
        """Load weights here, so loading time is not counted as inference time."""

    def predict(self, clip, prompts=None):
        H, W = clip.hw
        out = np.zeros((len(clip), H, W), np.uint16)
        for t in range(len(clip)):
            image = clip.image(t)
            # Replace this toy rule with your network: it marks very green-yellow pixels as one "cluster".
            r, g, b = (image[..., i].astype(int) for i in range(3))
            mask = (g > r + 20) & (g > b + 40) & (g > 255 * self.threshold)
            out[t][mask] = 1
        return out
