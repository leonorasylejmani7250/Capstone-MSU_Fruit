#@title Tools: labels in from Roboflow / CVAT / Label Studio, labels out to YOLO  { display-mode: "form" }
def _guess_source(filename, fallback):
    """Which video a labelled frame came from, read off its file name, so frames of one video never end up on
    both sides of the train / val split. Handles Roboflow ('clip_mp4-0012_jpg.rf.3f2a….jpg'), Label Studio
    ('1a2b3c4d-clip_0012.jpg') and '<video>_000123' / '<video>-frame-12' names. Bare numbers → `fallback`."""
    stem = re.sub(r"\.rf\.[0-9a-fA-F]+", "", Path(filename).name)
    stem = re.sub(r"_(jpe?g|png|bmp)$", "", Path(stem).stem, flags=re.I)
    stem = re.sub(r"^[0-9a-f]{8}-", "", stem)
    src = re.sub(r"[-_ .]*(frame|img|image)?[-_ .]*\d+$", "", stem, flags=re.I)
    src = re.sub(r"[_.-](mp4|mov|m4v|avi|mkv|mts|webm)$", "", src, flags=re.I)
    return _safe(src) or fallback

def _add_image_clip(name, img, ids, split, source, max_side, boxes=()):
    """Save one labelled image as a one-frame clip. `boxes` (x0, y0, x1, y1) are objects that came without an
    outline: SAM 2.1 draws their masks, and the clip stays a draft until a person checks it in label()."""
    h, w = ids.shape; sc = min(1.0, max_side / max(h, w))
    if sc < 1:
        size = (round(w * sc), round(h * sc))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        ids = cv2.resize(ids, size, interpolation=cv2.INTER_NEAREST)
        boxes = [[v * sc for v in b] for b in boxes]
    (IMG / name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(IMG / name / "00000.jpg", quality=95)
    meta = dict(source=source, status="done")
    if split in ("train", "val"):
        meta["split"] = split
    if len(boxes):
        k0 = int(ids.max())
        prompts = [dict(obj=k0 + i, frame=0, box=[float(v) for v in b]) for i, b in enumerate(boxes, 1)]
        best = np.where(ids > 0, 1000, 0).astype(np.float16)[None]        # outlined objects keep their pixels
        ids = Sam2Session(load_sam2(), Clip(name)).add(prompts, ids[None].astype(np.uint16), best)[0]
        _save_prompts(name, prompts)              # lets label() refine them with clicks or a new box
        meta["status"] = "draft"
    save_idmap(ANN / name / "00000.png", ids)
    set_meta(name, **meta)
    return meta["status"]

def _finish_import(source, made, split, drafted, extra=""):
    if split == "auto":
        print("split:", split_clips(VAL_FRACTION))
    print(f"{source}: {made} image(s) imported" + (f" as '{split}'" if split != "auto" else "") + extra)
    if drafted:
        print(f"  {drafted} image(s) had boxes only: SAM 2.1 drew those masks, so they are drafts. Check them in "
              "label() and mark each one done, or accept them unchecked: mark_done([...]) in the notebook, "
              "`python fruit.py approve --all-drafts` on the command line.")

def import_coco(json_path, images_dir=None, split="auto", categories=None, source=None, max_side=1920):
    """COCO segmentation export (Roboflow 'COCO Segmentation', CVAT 'COCO 1.0', Label Studio 'COCO').
    Every image becomes a one-frame clip. `categories`: names to keep (None = all, each instance one object).
    split="auto" groups images by the video they came from (read from the file name) and holds whole videos
    out for val. Do not reuse Roboflow's own train/valid split: it mixes frames of one video across both.
    Objects with a box and no outline get a SAM 2.1 mask and leave the image as a draft to check."""
    json_path = Path(json_path); images_dir = Path(images_dir or json_path.parent)
    coco = json.loads(json_path.read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    keep = {i for i, n in names.items() if categories is None or n in categories}
    per_image = {}
    for a in coco["annotations"]:
        if a["category_id"] in keep and not a.get("iscrowd", 0):
            per_image.setdefault(a["image_id"], []).append(a)
    source = source or json_path.parent.name
    made = empty = missing = drafted = 0
    for im in tqdm(coco["images"], desc=f"import {source}"):
        src = images_dir / im["file_name"]
        if not src.exists():
            missing += 1; continue
        img = np.asarray(Image.open(src).convert("RGB")); h, w = img.shape[:2]
        sx, sy = w / im.get("width", w), h / im.get("height", h)
        anns = sorted(per_image.get(im["id"], []), key=lambda a: -a.get("area", 0))     # big first, small on top
        if len(anns) > 255:
            print(f"  {im['file_name']}: {len(anns)} objects, keeping the 255 largest"); anns = anns[:255]
        ids, boxes, k = np.zeros((h, w), np.uint16), [], 0
        for a in anns:
            seg = a.get("segmentation")
            if isinstance(seg, list) and any(len(p) >= 6 for p in seg):     # polygons
                polys = [(np.asarray(p, np.float32).reshape(-1, 2) * (sx, sy)).round().astype(np.int32) for p in seg if len(p) >= 6]
                m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, polys, 1)
            elif isinstance(seg, dict):                                       # run-length masks
                from pycocotools import mask as _mu
                rle = _mu.frPyObjects(seg, seg["size"][0], seg["size"][1]) if isinstance(seg["counts"], list) else seg
                m = cv2.resize(_mu.decode(rle), (w, h), interpolation=cv2.INTER_NEAREST)
            else:                                                             # box only → SAM 2.1 draws it
                x, y, bw, bh = a["bbox"]
                boxes.append([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy]); continue
            k += 1; ids[m > 0] = k
        empty += int(k == 0 and not boxes)
        stem = Path(im["file_name"]).stem
        status = _add_image_clip(_safe(f"{source}_{stem}")[:90], img, ids, split, _guess_source(im["file_name"], source),
                                 max_side, boxes)
        made += 1; drafted += status == "draft"
    _finish_import(source, made, split, drafted, (f", {empty} with no objects" if empty else "")
                   + (f", {missing} listed but not found in {images_dir}" if missing else ""))

def import_yolo_seg(images_dir, labels_dir, split="auto", classes=None, source=None, max_side=1920):
    """YOLO labels, one .txt per image. Outline lines ('class x1 y1 x2 y2 …', 0-1 coordinates) become masks;
    box lines ('class cx cy w h') get a SAM 2.1 mask and leave the image as a draft to check.
    `classes`: class numbers to keep (None = all). split="auto": see import_coco."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    source = source or images_dir.parent.name or images_dir.name
    made = drafted = 0
    for src in tqdm(sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")), desc=f"import {source}"):
        txt = labels_dir / (src.stem + ".txt")
        img = np.asarray(Image.open(src).convert("RGB")); h, w = img.shape[:2]
        shapes, boxes = [], []
        for line in (txt.read_text().splitlines() if txt.exists() else []):
            v = line.split()
            if len(v) < 5 or (classes is not None and int(float(v[0])) not in classes): continue
            c = np.asarray(v[1:], np.float32)
            if len(c) == 4:
                cx, cy, bw, bh = c
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            else:
                shapes.append((c[: len(c) // 2 * 2].reshape(-1, 2) * (w, h)).round().astype(np.int32))
        shapes.sort(key=lambda p: -cv2.contourArea(p))
        ids = np.zeros((h, w), np.uint16)
        for k, poly in enumerate(shapes[:255], 1):
            m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [poly], 1); ids[m > 0] = k
        boxes = boxes[: max(0, 255 - min(len(shapes), 255))]
        status = _add_image_clip(_safe(f"{source}_{src.stem}")[:90], img, ids, split, _guess_source(src.name, source),
                                 max_side, boxes)
        made += 1; drafted += status == "draft"
    _finish_import(source, made, split, drafted)

def _one_polygon(mask):
    """Outline of a mask as ONE polygon, which is what YOLO labels need. An object cut into several pieces
    (by a leaf, a post, another cluster) keeps all of them: the pieces are joined by zero-width bridges."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = sorted((c.reshape(-1, 2) for c in cs if len(c) >= 3 and cv2.contourArea(c) >= 20),
                   key=lambda q: -cv2.contourArea(q))
    if not parts:
        return None
    poly = parts[0]
    for q in parts[1:]:
        d = ((poly[:, None, :].astype(np.int64) - q[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d.argmin()), d.shape)        # closest pair of points: bridge there and back
        poly = np.concatenate([poly[:i + 1], q[j:], q[:j + 1], poly[i:]])
    return poly

def export_yolo_seg(out_dir=None, every=2, name=None):
    """Write the labelled clips as a YOLO segmentation dataset (same train / val split), ready for
    `YOLO('yolo26n-seg.pt').train(data=<returned yaml>)`. `every`: keep every n-th frame (neighbours are near-copies)."""
    out = Path(out_dir or ROOT / "yolo_dataset"); n = {"train": 0, "val": 0}
    for split in n:
        for sub in ("images", "labels"):
            shutil.rmtree(out / sub / split, ignore_errors=True); (out / sub / split).mkdir(parents=True)
        for clip in get_clips(split, labelled=True):
            lab = clip.labels(); h, w = clip.hw
            for t in range(0, len(clip), every):
                lines = []
                for k in np.unique(lab[t]):
                    if k == 0: continue
                    poly = _one_polygon(lab[t] == k)
                    if poly is None: continue
                    pts = (poly / (w, h)).clip(0, 1)
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in pts.ravel()))
                stem = f"{clip.name}_{t:05d}"
                shutil.copy(clip.frames[t], out / "images" / split / f"{stem}.jpg")
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n[split] += 1
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: {name or OBJECT_NAME}\n")
    print(f"YOLO dataset: {n['train']} train + {n['val']} val images → {yaml}")
    return yaml

