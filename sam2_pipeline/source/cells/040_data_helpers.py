#@title C · Tools: clips, label files, overlays  { display-mode: "form" }
# One format for everything, the same DAVIS layout Meta's SAM 2 training code reads:
#   dataset/JPEGImages/<clip>/00000.jpg ...      frames
#   dataset/Annotations/<clip>/00000.png ...     labels: pixel value = object id, 0 = background
# Model predictions are stored the same way under predictions/<model>/<clip>/, which is what
# lets one scoring function grade any model.

def _davis_palette():
    pal = np.zeros((256, 3), np.uint8)
    for i in range(256):
        c = i
        for j in range(8):
            pal[i, 0] |= ((c >> 0) & 1) << (7 - j)
            pal[i, 1] |= ((c >> 1) & 1) << (7 - j)
            pal[i, 2] |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
    return pal

PALETTE = _davis_palette()
# Overlay colours chosen to stand out against green leaves (no greens).
OVERLAY_COLORS = np.array([(255, 0, 255), (0, 255, 255), (255, 140, 0), (255, 235, 0), (255, 40, 40),
                           (40, 110, 255), (255, 150, 200), (255, 255, 255), (160, 60, 255),
                           (0, 190, 255), (255, 110, 90), (200, 170, 255)], np.uint8)

def save_idmap(path, idmap):
    """Write an instance-id image. Palette PNG up to 255 objects (DAVIS style), 16-bit PNG beyond."""
    idmap = np.asarray(idmap)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if idmap.size == 0 or int(idmap.max()) <= 255:
        im = Image.frombytes("P", (idmap.shape[1], idmap.shape[0]), idmap.astype(np.uint8).tobytes())
        im.putpalette(PALETTE.flatten().tolist())
    else:
        im = Image.fromarray(idmap.astype(np.uint16))
    im.save(path)

def load_idmap(path):
    im = Image.open(path)
    if im.mode not in ("P", "L", "1", "I;16", "I"):
        raise ValueError(f"{path}: expected a PNG whose pixel values are object ids, got a {im.mode} image")
    return np.asarray(im).astype(np.uint16)

@dataclass
class Clip:
    name: str
    root: object = None                      # folder holding the frames folder; None = the dataset
    @property
    def dir(self): return (Path(self.root) if self.root else IMG) / self.name
    @functools.cached_property
    def frames(self): return sorted(self.dir.glob("*.jpg"))      # listed once per Clip object
    def __len__(self): return len(self.frames)
    @functools.cached_property
    def hw(self):
        w, h = Image.open(self.frames[0]).size
        return h, w
    def image(self, i): return np.asarray(Image.open(self.frames[i]).convert("RGB"))
    def labels(self, root=None):
        """All label frames as one [T, H, W] array, or None if the clip is not fully labelled."""
        d = (ANN if root is None else Path(root)) / self.name
        files = [d / (f.stem + ".png") for f in self.frames]
        if not files or not all(f.exists() for f in files):
            return None
        return np.stack([load_idmap(f) for f in files])
    def save_labels(self, idmaps, root=None, only=None):
        """Write one PNG per frame (files are overwritten in place). only = frame numbers to write."""
        d = (ANN if root is None else Path(root)) / self.name
        keep = {f.stem + ".png" for f in self.frames}
        for t, (f, m) in enumerate(zip(self.frames, idmaps)):
            if only is None or t in only:
                save_idmap(d / (f.stem + ".png"), m)
        for stale in d.glob("*.png"):
            if stale.name not in keep: stale.unlink()

# ---- the clip table (one small JSON next to the data keeps split + origin of every clip) ----
def _load_meta(): return json.loads(_META.read_text()) if _META.exists() else {}
def _save_meta(m): _META.write_text(json.dumps(m, indent=1))
def set_meta(clip, **kw):
    m = _load_meta(); m.setdefault(clip, {}).update(kw); _save_meta(m)

_count_cache = {}
def _count_objects(files):
    """Distinct object ids across label files; re-read only when a file has changed."""
    key = tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    if _count_cache.get(files[0].parent if files else None, (None,))[0] != key:
        objs = set()
        for f in files: objs |= set(np.unique(load_idmap(f)).tolist())
        _count_cache[files[0].parent if files else None] = (key, len(objs - {0}))
    return _count_cache[files[0].parent if files else None][1]

def clips_table():
    """Every clip on disk. labels: none | draft (not yet approved by a person) | done (approved).
    Only 'done' clips are used for training and scoring."""
    meta, rows = _load_meta(), []
    for d in sorted(p for p in IMG.iterdir() if p.is_dir()):
        c = Clip(d.name); info = meta.get(c.name, {}); n = len(c)
        labs = sorted((ANN / c.name).glob("*.png")) if (ANN / c.name).exists() else []
        status = (info.get("status") or "draft") if (n and len(labs) == n) else "none"
        rows.append(dict(clip=c.name, split=info.get("split", ""), frames=n, labels=status,
                         labelled=status == "done", objects=_count_objects(labs) if labs else 0,
                         source=info.get("source", ""), start_s=info.get("start_s", "")))
    t = pd.DataFrame(rows, columns=["clip", "split", "frames", "labels", "labelled", "objects", "source", "start_s"])
    return t.astype({"labelled": bool, "frames": int, "objects": int})          # right types even when empty

def get_clips(split=None, labelled=None):
    t = clips_table()
    if split is not None: t = t[t["split"] == split]
    if labelled is not None: t = t[t["labelled"] == labelled]
    return [Clip(n) for n in t["clip"]]

# ---- drawing -----------------------------------------------------------------------------------
def overlay(img, idmap, alpha=0.45, ids=True, outline=True):
    """Colour each object on top of the frame. Returns an RGB uint8 image."""
    out = img.copy()
    present = np.unique(idmap); present = present[present != 0]
    if len(present) == 0:
        return out
    color = OVERLAY_COLORS[(idmap.astype(np.int64) - 1) % len(OVERLAY_COLORS)]
    m = idmap > 0
    out[m] = (out[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    if outline:
        k, f = np.ones((3, 3), np.uint8), idmap.astype(np.float32)
        edge = cv2.dilate(f, k) != cv2.erode(f, k)
        out[edge & m] = color[edge & m]          # solid object colour just inside the boundary
        out[edge & ~m] = 255                     # white line just outside it
    if ids:
        s = max(0.45, min(out.shape[:2]) / 900)
        ys, xs = np.nonzero(m); lab = idmap[ys, xs].astype(np.int64)
        cnt = np.bincount(lab); cy = np.bincount(lab, ys); cx = np.bincount(lab, xs)
        for k_ in present:
            p = (int(cx[k_] / cnt[k_]) - int(8 * s), int(cy[k_] / cnt[k_]) + int(8 * s))
            cv2.putText(out, str(int(k_)), p, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), max(2, int(4 * s)), cv2.LINE_AA)
            cv2.putText(out, str(int(k_)), p, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
    return out

def show_clip(clip, idmaps=None, n=4, title=None, width=16):
    """Contact sheet of n evenly spaced frames. idmaps defaults to the clip's saved labels."""
    clip = Clip(clip) if isinstance(clip, str) else clip
    if idmaps is None: idmaps = clip.labels()
    T = len(clip); picks = sorted(set(np.linspace(0, T - 1, min(n, T)).round().astype(int).tolist()))
    h, w = clip.hw
    fig, axes = plt.subplots(1, len(picks), figsize=(width, width / len(picks) * h / w + 0.4), squeeze=False)
    for ax, i in zip(axes[0], picks):
        im = clip.image(i)
        ax.imshow(overlay(im, idmaps[i]) if idmaps is not None else im)
        ax.set_title(f"frame {i}", fontsize=9); ax.axis("off")
    fig.suptitle(title or clip.name, fontsize=11); plt.tight_layout(); plt.show()

def write_video(frames, path, fps=6):
    """Write RGB frames (a list or a generator) to an .mp4 that plays in the browser (H.264 if ffmpeg is there)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp, vw = path.with_suffix(".tmp.mp4"), None
    for f in frames:
        h, w = f.shape[0] - f.shape[0] % 2, f.shape[1] - f.shape[1] % 2
        if vw is None:
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vw.write(cv2.cvtColor(np.ascontiguousarray(f[:h, :w]), cv2.COLOR_RGB2BGR))
    if vw is None:
        raise ValueError("no frames to write")
    vw.release()
    if shutil.which("ffmpeg") and subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)]).returncode == 0:
        tmp.unlink()
    else:
        tmp.replace(path)
    return path

# ---- videos → clips → train / val ---------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".webm"}

def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dict(fps=float(fps), frames=n, width=w, height=h, seconds=(n / fps if fps else 0.0))

def list_videos():
    rows = []
    for p in sorted(VIDEOS.iterdir()):
        if p.suffix.lower() in VIDEO_EXTS:
            info = probe_video(p)
            rows.append(dict(video=p.name, MB=round(p.stat().st_size / 2**20, 1), **(info or {"fps": None})))
    return pd.DataFrame(rows)

def cut_clips(video, clips_per_video=6, clip_frames=16, clip_fps=6.0, max_side=1280, starts=None):
    """Extract evenly spaced, non-overlapping clips from one video. Safe to re-run: existing clips are kept."""
    video = Path(video); info = probe_video(video)
    if not info or not info["fps"] or not info["frames"]:
        print(f"  ✗ cannot read {video.name}"); return []
    stride = max(1, round(info["fps"] / clip_fps))          # keep every stride-th frame
    span = clip_frames * stride / info["fps"]               # seconds of video one clip covers
    dur = info["seconds"]
    if starts is None:
        k = min(clips_per_video, max(1, int(dur // span)))
        starts = [(i + 0.5) * dur / k - span / 2 for i in range(k)]
    taken = {n: (m.get("start_s") or 0, (m.get("start_s") or 0) + m.get("span_s", 0)) for n, m in _load_meta().items()
             if m.get("source") == video.name and (IMG / n).exists()}
    made, cap = [], cv2.VideoCapture(str(video))
    for s in starts:
        s = max(0.0, min(s, max(0.0, dur - span)))
        name = f"{_safe(video.stem)}_t{int(round(s * 1000)):07d}"
        out = IMG / name
        if name in taken:
            made.append(name); continue
        clash = [n for n, (a, b) in taken.items() if s < b and a < s + span]
        if clash:                      # never create a clip that shares frames with an existing one
            print(f"  · {name} would overlap {clash[0]}, skipped"); continue
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        frames, i = [], 0
        while len(frames) < clip_frames and cap.grab():
            if i % stride == 0:
                ok, bgr = cap.retrieve()
                if not ok: break
                h, w = bgr.shape[:2]; sc = max_side / max(h, w)
                if sc < 1:
                    bgr = cv2.resize(bgr, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
                frames.append(bgr)
            i += 1
        if len(frames) < clip_frames:
            print(f"  ✗ {name}: only {len(frames)} of {clip_frames} frames readable, skipped"); continue
        out.mkdir(parents=True, exist_ok=True)
        for j, f in enumerate(frames):
            cv2.imwrite(str(out / f"{j:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        set_meta(name, source=video.name, start_s=round(s, 3), span_s=round(span, 3),
                 fps=round(info["fps"] / stride, 2))
        taken[name] = (s, s + span); made.append(name)
    cap.release()
    return made

def split_clips(val_fraction=0.25, reset=False):
    """Give every clip without a split a 'train' or 'val' tag.
    3+ source videos: whole videos are held out. Fewer: the last clips (in time) of each video are held out.
    Clips that already have a split keep it, so adding videos later never moves a val clip into train."""
    meta = _load_meta()
    names = [d.name for d in sorted(IMG.iterdir()) if d.is_dir()]
    for n in names:
        meta.setdefault(n, {})
        if reset: meta[n]["split"] = ""
    by_src = {}
    for n in names:
        by_src.setdefault(meta[n].get("source") or n, []).append(n)
    count = lambda s: sum(1 for n in names if meta[n].get("split") == s)
    whole_videos = len(by_src) >= 3                 # a source = one video (clips cut here, or imported frames)
    for src in sorted(by_src):
        cl = sorted(by_src[src], key=lambda n: (meta[n].get("start_s") or 0, n))
        todo = [n for n in cl if not meta[n].get("split")]
        if not todo: continue
        already = {meta[n]["split"] for n in cl if meta[n].get("split")}
        if whole_videos and len(already) <= 1:
            s = already.pop() if already else (
                "val" if count("val") < val_fraction * (count("train") + count("val") + len(todo)) else "train")
            for n in todo: meta[n]["split"] = s
        else:
            k = max(1, round(len(cl) * val_fraction)) if len(cl) > 1 else 0
            for i, n in enumerate(cl):
                if n in todo: meta[n]["split"] = "val" if i >= len(cl) - k else "train"
    _save_meta(meta)
    return "whole videos held out" if whole_videos else "last clips of each video held out"

def set_split(clip, split):
    assert split in ("train", "val"); set_meta(clip, split=split)

def delete_clip(clip):
    for d in (IMG / clip, ANN / clip):
        if d.exists(): shutil.rmtree(d)
    m = _load_meta(); m.pop(clip, None); _save_meta(m)

VAL_FRACTION = 0.25          # share of clips (or of whole videos) held out for scoring; Step 2 can change it

def dataset_report(show_examples=True):
    """What is usable for training and scoring, plus the problems worth fixing first."""
    t = clips_table()
    display(t)
    ok = t.loc[t["labelled"].astype(bool)]
    n = {s: int((ok["split"] == s).sum()) for s in ("train", "val")}
    print(f"usable (labelled and marked done): {n['train']} train clip(s), {n['val']} val clip(s), "
          f"{int(ok['objects'].sum())} labelled objects in total")
    for msg, bad in (("no labelled val clips: nothing can be scored yet", n["val"] == 0),
                     ("no labelled train clips: nothing to fine-tune on yet", n["train"] == 0),
                     (f"{int((t['labels'] == 'draft').sum())} clip(s) are still drafts: open them in label() and mark them done",
                      bool((t["labels"] == "draft").any())),
                     ("fewer than 3 source videos: train and val clips come from the same video, so scores will flatter "
                      "every model. Film val footage on a different row or day when you can.",
                      0 < t["source"].nunique() < 3 and not DEMO),
                     ("under 10 labelled train clips is enough to test the plumbing, not to expect a real gain",
                      0 < n["train"] < 10 and not DEMO)):
        if bad: print("⚠️ ", msg)
    if show_examples and _in_notebook():
        for c in get_clips("train", labelled=True)[:1] + get_clips("val", labelled=True)[:1]:
            show_clip(c, title=f"{c.name}  ({_load_meta()[c.name].get('split')})")
    return t

# ---- public sample data (Embrapa WGISD, CC BY-NC 4.0): real grape photos with hand-drawn cluster masks -------
_WGISD = "https://raw.githubusercontent.com/thsant/wgisd/master/data/"
DEMO_IMAGES = ["CDY_2015", "CDY_2017", "CDY_2018", "CDY_2019", "SVB_1931", "SVB_1935", "SVB_1937", "SVB_1938"]

def make_demo_clips(names=None, frames=12, fps=6, hw=(640, 960)):
    """Turn WGISD photos into short panning clips with exact labels, through the same code path as real videos
    (written as .mp4 into videos/, cut by cut_clips). Gives every step something to run on before you have
    labelled anything. Returns the clip names; existing demo clips are kept."""
    raw = BASE / "demo_download"; raw.mkdir(parents=True, exist_ok=True)
    made = []
    for name in names or DEMO_IMAGES:
        clip_name = f"demo_{name}_t0000000"
        if (ANN / clip_name).exists():
            made.append(clip_name); continue
        for ext in ("jpg", "npz"):
            f = raw / f"{name}.{ext}"
            if not f.exists():
                torch.hub.download_url_to_file(_WGISD + f.name, str(f), progress=False)
        img = np.asarray(Image.open(raw / f"{name}.jpg").convert("RGB"))
        masks = np.load(raw / f"{name}.npz")["arr_0"]                        # [H, W, N], one layer per cluster
        ids = np.zeros(img.shape[:2], np.uint16)
        for k in np.argsort(-masks.reshape(-1, masks.shape[2]).sum(0)):      # large first, small ones on top
            ids[masks[..., k] > 0] = k + 1
        H, W = ids.shape; oh, ow = hw
        win_w = int(W * 0.6); win_h = int(win_w * oh / ow)
        clip_frames, labels = [], []
        for i in range(frames):                                              # pan left → right with a gentle bob
            a = i / (frames - 1)
            x0 = int(a * (W - win_w)); y0 = int((H - win_h) * (0.5 + 0.25 * math.sin(a * math.pi)))
            clip_frames.append(cv2.resize(img[y0:y0 + win_h, x0:x0 + win_w], (ow, oh), interpolation=cv2.INTER_AREA))
            labels.append(cv2.resize(ids[y0:y0 + win_h, x0:x0 + win_w], (ow, oh), interpolation=cv2.INTER_NEAREST))
        vid = VIDEOS / f"demo_{name}.mp4"
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        for f in clip_frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        got = cut_clips(vid, 1, frames, fps, max(hw))                        # the same path your videos take
        assert got == [clip_name] and len(Clip(clip_name)) == frames, f"demo clip {name} did not extract cleanly"
        Clip(clip_name).save_labels(np.stack(labels))
        set_meta(clip_name, status="done")
        made.append(clip_name)
    return made
