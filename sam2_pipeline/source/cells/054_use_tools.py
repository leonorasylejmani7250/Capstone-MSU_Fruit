#@title Tools: side-by-side comparisons and running a pipeline over long video  { display-mode: "form" }
def compare(clip, names, frames=4, video=True):
    """Side by side: the labels, then each named model's saved predictions, on the same frames."""
    clip = Clip(clip) if isinstance(clip, str) else clip
    rows = [("labels (ground truth)", clip.labels())] + [(n, clip.labels(root=PRED / _safe(n))) for n in names]
    rows = [(n, a) for n, a in rows if a is not None]
    picks = np.linspace(0, len(clip) - 1, min(frames, len(clip))).round().astype(int)
    h, w = clip.hw
    fig, ax = plt.subplots(len(rows), len(picks), figsize=(4.2 * len(picks), 4.2 * h / w * len(rows) + 0.3), squeeze=False)
    for r, (n, a) in enumerate(rows):
        for c, t in enumerate(picks):
            ax[r, c].imshow(overlay(clip.image(t), a[t], ids=False)); ax[r, c].axis("off")
            ax[r, c].set_title(f"{n} · frame {t}" if c == 0 else f"frame {t}", fontsize=8, loc="left")
    plt.tight_layout()
    if _in_notebook():
        plt.show()
    else:
        plt.savefig(RESULTS / f"compare_{clip.name}.png", dpi=110); plt.close()
    if video and len(clip) > 1:
        def tiles():
            for t in range(len(clip)):
                row = []
                for n, a in rows:
                    im = overlay(clip.image(t), a[t], ids=False)
                    im = cv2.resize(im, (480, round(480 * h / w)), interpolation=cv2.INTER_AREA)
                    cv2.putText(im, n[:46], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(im, n[:46], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    row.append(im)
                yield np.concatenate(row, axis=1)
        out = write_video(tiles(), RESULTS / f"compare_{clip.name}.mp4", fps=4)
        if _in_notebook():
            display(Video(str(out), embed=True, html_attributes="controls loop"))
        else:
            print("comparison video:", out)

def run_on_video(model, video, start_s=0, seconds=0, fps=6, max_side=1280, chunk=24, min_track_frames=2):
    """Run an automatic model over a video (seconds=0: to its end) in chunks that share one frame, and count the
    clusters: every object keeps one id through the whole video. A detector → SAM 2.1 pipeline hands each
    chunk the objects it is following; any other model has its ids matched on the shared frame.
    Writes results/sam2/: the overlay video, per-frame counts and one row per object, and for a whole video one
    line in results/cluster_counts.csv. Returns (overlay video path, per-frame counts, per-object tracks)."""
    video = Path(video); info = probe_video(video)
    assert info and info["fps"], f"cannot read {video}"
    stride = max(1, round(info["fps"] / fps))
    whole = not start_s and (not seconds or seconds >= info["seconds"] - 0.5)
    work = BASE / "video_runs" / f"{_safe(video.stem)}_{int(start_s)}s_{int(seconds)}s"
    shutil.rmtree(work, ignore_errors=True); (work / "all").mkdir(parents=True)
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    n, i, limit = 0, 0, max(1, int(seconds * info["fps"] / stride)) if seconds else 10**9
    while n < limit and cap.grab():
        if i % stride == 0:
            ok, bgr = cap.retrieve()
            if not ok: break
            sc = max_side / max(bgr.shape[:2])
            if sc < 1: bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(work / "all" / f"{n:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]); n += 1
        i += 1
    cap.release()
    assert n > 0, "no frames could be read from that part of the video"
    model.prepare()
    det = getattr(model, "detector", model)
    auto = getattr(det, "imgsz", None) == "auto"
    if auto:                                                  # one detector input size for the whole stretch
        det.imgsz = det.fit([cv2.cvtColor(cv2.imread(str(work / "all" / f"{t:05d}.jpg")), cv2.COLOR_BGR2RGB)
                             for t in sorted({int(t) for t in np.linspace(0, n - 1, 5)})])
        print(f"detector input size: {det.imgsz} px, where it is most sure (tried {', '.join(str(v) for v in det.tried['imgsz'])})")
    carries = isinstance(model, DetectThenSAM2)
    tracks, per_frame, state = {}, [], dict(last=None, next_id=1)

    def overlays():
        start = 0
        while True:
            stop = min(start + chunk, n)
            cdir = work / f"chunk_{start:05d}"; cdir.mkdir()
            for j, t in enumerate(range(start, stop)):
                try:
                    os.symlink(work / "all" / f"{t:05d}.jpg", cdir / f"{j:05d}.jpg")
                except OSError:                       # Windows without Developer Mode: copy instead
                    shutil.copy2(work / "all" / f"{t:05d}.jpg", cdir / f"{j:05d}.jpg")
            clip = Clip(cdir.name, root=work)
            if carries:                               # the objects being followed go on under their ids
                ids = np.asarray(model.predict(clip, carry=state["last"], first_id=state["next_id"])).astype(np.int64)
                state["next_id"] = max(state["next_id"], int(ids.max()) + 1)
            else:
                ids = np.asarray(model.predict(clip)).astype(np.int64)
                lut = np.zeros(int(ids.max()) + 1, np.int64)
                if state["last"] is not None:                     # same frame seen by both chunks: carry ids over
                    old_ids, new_ids, iou, _, _ = iou_matrix(state["last"], ids[0])
                    if iou.size:
                        r, c = linear_sum_assignment(-iou)
                        for a, b in zip(r, c):
                            if iou[a, b] >= 0.5: lut[new_ids[b]] = old_ids[a]
                for k in np.unique(ids):
                    if k and lut[k] == 0:
                        lut[k] = state["next_id"]; state["next_id"] += 1
                ids = lut[ids]
            for j in range(0 if state["last"] is None else 1, len(ids)):
                t = start + j
                k, cnt = np.unique(ids[j], return_counts=True); k = k[(k != 0) & (cnt >= MIN_OBJECT_AREA)]
                for o in k:
                    tr = tracks.setdefault(int(o), dict(object=int(o), first_frame=t, last_frame=t, frames=0))
                    tr["last_frame"] = t; tr["frames"] += 1
                per_frame.append(dict(frame=t, second=round(start_s + t * stride / info["fps"], 2), objects_in_view=len(k),
                                      clusters_so_far=sum(1 for tr in tracks.values() if tr["frames"] >= min_track_frames)))
                im = overlay(clip.image(j), ids[j])
                im = cv2.resize(im, (960, round(960 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
                for colour, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                    cv2.putText(im, f"in view {len(k)} · clusters so far {per_frame[-1]['clusters_so_far']}", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, width, cv2.LINE_AA)
                yield im
            state["last"] = ids[-1]
            shutil.rmtree(cdir, ignore_errors=True)
            if stop >= n: break
            start = stop - 1

    (RESULTS / "sam2").mkdir(parents=True, exist_ok=True)           # the SAM 2.1 pipeline's own video runs
    try:
        out = write_video(overlays(), RESULTS / "sam2" / f"overlay_{_safe(video.stem)}_{int(start_s)}s.mp4", fps=info["fps"] / stride)
    finally:
        if auto: det.imgsz = "auto"
        shutil.rmtree(work, ignore_errors=True)
    counts = pd.DataFrame(per_frame)
    tr = pd.DataFrame(list(tracks.values()), columns=["object", "first_frame", "last_frame", "frames"])
    tr = tr[tr["frames"] >= min_track_frames].reset_index(drop=True)
    counts.to_csv(out.with_suffix(".counts.csv"), index=False); tr.to_csv(out.with_suffix(".tracks.csv"), index=False)
    print(f"{n} frames, {n * stride / info['fps']:.0f} s · {counts['objects_in_view'].mean():.1f} clusters in view on average · "
          f"{len(tr)} clusters in all (each followed for at least {min_track_frames} frames)\noverlay: {out}")
    if not counts["objects_in_view"].sum():
        print("⚠️  nothing found in this stretch. " + NOTHING_FOUND)
    if whole:
        f = record_count(video, model_label(model), len(tr), n * stride / info["fps"])
        print(f"total for this video: {len(tr)} clusters, written to {f}")
    return out, counts, tr

COUNT_COLUMNS = ["video", "model", "clusters", "seconds", "pipeline", "written"]

def record_count(video, model, clusters, seconds, pipeline="sam2"):
    """One line per counted video in results/cluster_counts.csv, the file every pipeline adds its totals to."""
    import csv
    f = RESULTS / "cluster_counts.csv"; f.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    if not f.exists() or not f.stat().st_size:
        w.writerow(COUNT_COLUMNS)
    w.writerow([video_key(video), model, int(clusters), round(float(seconds), 1), pipeline, time.strftime("%Y-%m-%d %H:%M")])
    with open(f, "a", encoding="utf-8") as h:              # one short write: safe when several jobs finish together
        h.write(buf.getvalue())
    return f

def counted(model):
    """The videos this model has a total for already."""
    f = RESULTS / "cluster_counts.csv"
    return set(pd.read_csv(f).query("model == @model")["video"]) if f.exists() and f.stat().st_size else set()

def count_table():
    """The total of every model for every video (the newest run of each), from results/cluster_counts.csv:
    one row per video, one column per model. Saved as results/cluster_counts_by_video.csv."""
    f = RESULTS / "cluster_counts.csv"
    if not f.exists():
        print(f"No counts yet: {f} is written by every whole-video run (fruit.py video, video.py, run.py video)."); return None
    t = pd.read_csv(f).drop_duplicates(["video", "model"], keep="last")
    table = t.pivot(index="video", columns="model", values="clusters")
    table = table[sorted(table.columns)].astype("Int64")
    table.to_csv(f.with_name("cluster_counts_by_video.csv"))
    print(f"clusters counted in each video, by model · {f.with_name('cluster_counts_by_video.csv')}")
    print(table.to_string(na_rep="-"))
    return table

def _short(p):
    """A path as it can be typed on the command line again: inside the project folder or the repository, relative to it."""
    for base in (ROOT, REPO_DIR):
        try:
            return str(Path(p).resolve().relative_to(base))
        except (ValueError, TypeError):
            pass
    return str(Path(p).resolve())

def check_detectors(video, detectors, frames=6, conf=SURE, max_side=1280):
    """Why does a model find nothing in my video? Runs every detector ({name: UltralyticsModel}) on a few frames
    spread over the video, at every input size, and says which detector and size to use, or that none works.
    Frames are shrunk to `max_side` first, as `clips` and `video` do. Writes results/check/<video>/: check.csv
    and one picture per detector (its boxes on the frames, at its best size). Returns the table."""
    video = Path(video); info = probe_video(video)
    assert info and info["fps"], f"cannot read {video}"
    cap, imgs = cv2.VideoCapture(str(video)), []
    for sec in np.linspace(0, info["seconds"], frames + 2)[1:-1]:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000); ok, bgr = cap.read()
        if not ok: continue
        sc = max_side / max(bgr.shape[:2]) if max_side else 1
        if sc < 1: bgr = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    assert imgs, f"no frames could be read from {video}"
    out = RESULTS / "check" / _safe(video.stem)
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    print(f"{video.name}: {info['width']}×{info['height']}, {info['seconds']:.0f} s · {len(imgs)} frames, "
          f"seen at {imgs[0].shape[1]}×{imgs[0].shape[0]} · a box counts from a score of {conf:g}")
    tables, best = [], {}
    for name, det in detectors.items():
        try:
            det.prepare()
            t = det.sweep(imgs, conf=conf)
            b = best[name] = t.loc[(t["sure"] if t["sure"].max() > 0 else t["top_score"]).idxmax()]
            tiles = []
            for im in imgs:                                   # the picture: this detector's boxes at its best size
                boxes, scores, _ = det.detect(im, conf=conf, imgsz=int(b["imgsz"]))
                tile = np.ascontiguousarray(im[..., ::-1])
                for (x0, y0, x1, y1), s_ in zip(boxes.round().astype(int), scores):
                    cv2.rectangle(tile, (x0, y0), (x1, y1), (0, 230, 255), 3)
                    cv2.putText(tile, f"{s_:.2f}", (x0 + 4, max(y0 + 24, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(tile, f"{s_:.2f}", (x0 + 4, max(y0 + 24, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)
                tiles.append(cv2.resize(tile, (640, round(640 * tile.shape[0] / tile.shape[1])), interpolation=cv2.INTER_AREA))
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
            cv2.imwrite(str(out / f"{_safe(name)}_{int(b['imgsz'])}px.jpg"),
                        np.concatenate([np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)]))
            tables.append(t.assign(detector=name))
        except Exception as e:                                # one detector that cannot load does not stop the others
            print(f"✗ {name}: {type(e).__name__}: {str(e)[:200]}")
    if not tables:
        raise RuntimeError("no detector could be run")
    table = pd.concat(tables, ignore_index=True)[["detector", "imgsz", "boxes_per_frame", "top_score", "box_height_px", "sure"]]
    table.to_csv(out / "check.csv", index=False)
    show = table.assign(cell=[f"{b_:.1f} ({s_:.2f})" for b_, s_ in zip(table["boxes_per_frame"], table["top_score"])])
    print("\nboxes per frame (top score), by the detector's input size:")
    print(show.pivot(index="imgsz", columns="detector", values="cell")[list(best)].to_string())
    trained = [k for k in best if not detectors[k].warn] or list(best)          # a test run never wins
    name = max(trained, key=lambda k: best[k]["sure"]); b = best[name]
    print(f"\npictures and table: {out}")
    if b["sure"] > 0:
        d = detectors[name]
        print(f"Most sure: {name} at {int(b['imgsz'])} px: {b['boxes_per_frame']:.1f} boxes per frame, top score {b['top_score']:.2f}.\n"
              f"Open {_safe(name)}_{int(b['imgsz'])}px.jpg. Boxes on the clusters: use it,\n"
              f"  python fruit.py video {_short(video)} --detector {_short(d.weights)}" + (f' --text "{d.text}"' if d.text else "") + "\n"
              "Boxes on leaves, or clusters without a box: label clips of this footage and fine-tune (LABELLING.md).")
    else:
        print(f"No detector is sure of anything in this video at any size (highest score {table['top_score'].max():.2f}; sure is "
              f"{max(conf, SURE):g} or more).\nThe footage is unlike the photos they were trained on: label clips of it and fine-tune (LABELLING.md).")
    return table
