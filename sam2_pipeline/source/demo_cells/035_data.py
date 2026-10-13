clips = F.make_demo_clips(["CDY_2017", "CDY_2019", "SVB_1935", "SVB_1938"])
VAL = "demo_CDY_2019_t0000000"                  # the held-out clip: never trained on, only scored
for c in clips:
    F.set_split(c, "val" if c == VAL else "train")
F.dataset_report(show_examples=False)
F.show_clip(VAL, title="Held-out clip with its true labels (each colour is one grape cluster)")
