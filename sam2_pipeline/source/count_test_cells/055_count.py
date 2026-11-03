# Step 4: run the detector and the detector → SAM 2.1 pipeline on every photo and count the clusters.
STILLS = F.ROOT / "stills"                       # one single-frame clip per photo
OVERLAYS = F.RESULTS / "overlays"
OVERLAYS.mkdir(parents=True, exist_ok=True)

def still_clip(photo):
    """Turn one photo into a one-frame clip the pipeline can read."""
    d = STILLS / photo.stem
    d.mkdir(parents=True, exist_ok=True)
    frame = d / "00000.jpg"
    if not frame.exists():
        shutil.copy2(photo, frame)
    return F.Clip(photo.stem, root=STILLS)

rows = []
for filename, truth in tqdm(GROUND_TRUTH):
    photo = IMAGES / filename
    if not photo.exists():
        rows.append(dict(image=filename, truth=truth, status="missing image"))
        continue
    try:
        clip = still_clip(photo)
        boxes, scores, _ = DETECTOR.detect(clip.image(0))        # the detector alone: one box per cluster it finds
        ids = PIPELINE.predict(clip)                             # detector boxes → SAM 2.1 masks: [1, H, W] object ids, 0 = background
        objects = np.unique(ids[0])
        n_sam2 = int((objects != 0).sum())
        Image.fromarray(F.overlay(clip.image(0), ids[0])).save(OVERLAYS / filename)
        rows.append(dict(image=filename, truth=truth, yolo=len(boxes), sam2=n_sam2, status="OK"))
    except Exception as e:
        rows.append(dict(image=filename, truth=truth, status=f"{type(e).__name__}: {e}"))

df = pd.DataFrame(rows, columns=["image", "truth", "yolo", "sam2", "status"])
df["yolo_error"] = df["yolo"] - df["truth"]
df["sam2_error"] = df["sam2"] - df["truth"]
df.to_csv(F.RESULTS / "count_test.csv", index=False)
display(df)
