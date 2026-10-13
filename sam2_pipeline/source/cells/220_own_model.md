## Appendix C · Adding your own model in code
A model is a class with one method. It receives a clip and returns one array of object numbers per frame.

```python
class MyModel(Model):
    tracks = False                     # True if the same cluster keeps the same number across frames

    def predict(self, clip, prompts=None):
        out = np.zeros((len(clip), *clip.hw), np.uint16)
        for t in range(len(clip)):
            image = clip.image(t)                      # RGB, uint8, [height, width, 3]
            masks = my_network(image)                  # your code → a list of boolean [height, width] masks
            for number, mask in enumerate(masks, 1):
                out[t][mask] = number
        return out

benchmark({"my model": MyModel()})
```

For a model that needs a hint per object (like SAM 2), set `needs_prompts = True` and `prompt = "box"` (or `"point"`, `"mask"`); `prompts` then arrives as a list of `{"obj", "frame", "box" | "points" | "mask"}`, and the model is scored on the prompted table.

Already wrapped: `SAM2Prompted(size, ckpt, prompt)`, `UltralyticsModel(weights, text)` (YOLO, YOLOE), `DetectThenSAM2(detector, ckpt)`, `FromFolder(folder)`, `SAM3Text(text)`. `clear_leaderboard()` starts the table over.

The same models can be described as plain data, which is how the HPCC jobs take them (`configs/lineup.yaml`, see the README):

```python
benchmark(load_lineup("/path/to/lineup.yaml"))
benchmark({"YOLOE": build_model({"type": "ultralytics", "weights": "yoloe-26m-seg.pt", "text": "grape cluster"})})
```

`examples/my_model.py` is a ready-to-edit template for your own class (`type: python` in a lineup file).
