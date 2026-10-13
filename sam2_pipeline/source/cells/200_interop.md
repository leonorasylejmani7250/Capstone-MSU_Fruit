## Appendix A · Labels from other tools, and labels out to YOLO
For labels you already have, or are making in Roboflow, CVAT or Label Studio. Each labelled image becomes a one-frame clip.

**Import a COCO segmentation export** (Roboflow "COCO Segmentation", CVAT "COCO 1.0", Label Studio "COCO"). Upload and unzip the export, then:

```python
import_coco("/content/export/train/_annotations.coco.json")
import_coco("/content/export/valid/_annotations.coco.json")
# keep only some classes:  import_coco(..., categories=["grape cluster"])
```

**Import YOLO labels** (`images/` and `labels/` folders; outline lines and box lines both work):

```python
import_yolo_seg("/content/export/train/images", "/content/export/train/labels")
```

What happens on import:
- **Outlines** (polygons or masks) are used as they are, and the image counts as labelled.
- **Boxes only** (YOLO detection labels, COCO boxes without a polygon) get a mask drawn by SAM 2.1 from the box. Those images stay *drafts*: open them in `label()`, fix what is wrong and mark them done, or `mark_done([...])` to accept them unchecked. Never put rectangles in as masks: SAM would learn to draw rectangles.
- **Train / val split** is `split="auto"` by default: the video each frame came from is read off its file name (Roboflow's `…_mp4-0012_jpg.rf.<hash>.jpg` style works), and whole videos are held out for val. Roboflow's own train/valid folders mix frames of one video on both sides, which inflates scores, so they are ignored. Pass `split="train"` or `split="val"` to force one.

Single images train SAM 2.1 as an image model and are scored on the *IoU on prompt frame* column and on the automatic table. The tracking columns (J&F) need real clips.

**Export everything labelled here to YOLO format** (same train / val split), for training a detector outside this notebook:

```python
data_yaml = export_yolo_seg(every=2)      # every 2nd frame; neighbouring frames are near-copies
```

Step 5b does this export and the YOLO training in one go. On the HPCC: `python fruit.py import-coco …`, `import-yolo …`, `export-yolo`.
