# Step 6: look at four photos: SAM 2.1's masks, each cluster numbered, with the three counts in the title.
import matplotlib.pyplot as plt

assert len(ok), "no photo was counted; see the status column above"
picks = ok.iloc[np.linspace(0, len(ok) - 1, min(4, len(ok))).round().astype(int)]
fig, axes = plt.subplots(1, len(picks), figsize=(18, 4.2), squeeze=False)
for ax, r in zip(axes[0], picks.itertuples()):
    ax.imshow(Image.open(OVERLAYS / r.image))
    ax.set_title(f"{r.image}\nby hand {r.truth} · YOLO26 {r.yolo} · SAM 2.1 {r.sam2}", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()
