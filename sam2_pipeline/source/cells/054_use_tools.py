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

def run_on_video(model, video, start_s=0, seconds=10, fps=6, max_side=1280, chunk=24, min_track_frames=2):
    """Run an automatic model over part of a video in overlapping chunks and stitch the object ids together.
    Returns (overlay video path, per-frame counts, per-object tracks)."""
    video = Path(video); info = probe_video(video)
    assert info and info["fps"], f"cannot read {video}"
    stride = max(1, round(info["fps"] / fps))
    work = BASE / "video_runs" / f"{_safe(video.stem)}_{int(start_s)}s_{int(seconds)}s"
    shutil.rmtree(work, ignore_errors=True); (work / "all").mkdir(parents=True)
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    n, i, limit = 0, 0, max(1, int(seconds * info["fps"] / stride))
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
                per_frame.append(dict(frame=t, second=round(start_s + t * stride / info["fps"], 2), objects_in_view=len(k)))
                for o in k:
                    tr = tracks.setdefault(int(o), dict(object=int(o), first_frame=t, last_frame=t, frames=0))
                    tr["last_frame"] = t; tr["frames"] += 1
                im = overlay(clip.image(j), ids[j])
                yield cv2.resize(im, (960, round(960 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
            state["last"] = ids[-1]
            if stop >= n: break
            start = stop - 1

    out = write_video(overlays(), RESULTS / f"overlay_{_safe(video.stem)}_{int(start_s)}s.mp4", fps=info["fps"] / stride)
    counts = pd.DataFrame(per_frame)
    tr = pd.DataFrame(list(tracks.values()), columns=["object", "first_frame", "last_frame", "frames"])
    tr = tr[tr["frames"] >= min_track_frames].reset_index(drop=True)
    counts.to_csv(out.with_suffix(".counts.csv"), index=False); tr.to_csv(out.with_suffix(".tracks.csv"), index=False)
    print(f"{n} frames · {counts['objects_in_view'].mean():.1f} objects in view on average · "
          f"{len(tr)} distinct objects seen in at least {min_track_frames} frames\noverlay: {out}")
    return out, counts, tr
