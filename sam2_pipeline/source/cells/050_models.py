#@title Tools: model wrappers and the leaderboard  { display-mode: "form" }
# A model is anything with predict(clip, prompts) → [T, H, W] array of object ids (0 = background).
# To add your own: subclass Model, or write PNGs with another program and use FromFolder.

class Model:
    needs_prompts = False   # True → scored on the prompted leaderboard; False → on the automatic one
    tracks = False          # True → the same object keeps the same id from frame to frame
    prompt = None           # for prompted models: "box" | "point" | "mask"
    def prepare(self):      # load weights here so loading time is not counted as inference time
        pass
    def predict(self, clip, prompts=None):
        raise NotImplementedError

class SAM2Prompted(Model):
    """SAM 2.1 pointed at each object once. ckpt=None → Meta's pretrained weights; or a fine-tuned .pt file."""
    needs_prompts, tracks = True, True
    def __init__(self, size=None, ckpt=None, prompt="box"):
        self.size, self.ckpt, self.prompt = size or SAM2_SIZE, ckpt, prompt
    def prepare(self):
        load_sam2(self.size, self.ckpt)
    def predict(self, clip, prompts=None):
        return sam2_track(load_sam2(self.size, self.ckpt), clip, prompts, backward=False)

def _ultralytics(text=False):
    """Import ultralytics, installing it (and its tracker / text-prompt extras) on first use."""
    need = [pkg for mod, pkg in (("ultralytics", "ultralytics"), ("lap", "lap>=0.5.12")) if _missing(mod)]
    if text and _missing("clip"):
        need.append("git+https://github.com/ultralytics/CLIP.git")          # text encoder for YOLOE prompts
    if need:
        print("installing:", ", ".join(need))
        sh([sys.executable, "-m", "pip", "install", "-q", *need])
    threads = torch.get_num_threads()
    import ultralytics
    torch.set_num_threads(threads)                # importing ultralytics lowers the CPU thread count; put it back
    return ultralytics

class UltralyticsModel(Model):
    """Any Ultralytics weights: a YOLO-seg model you trained, or YOLOE with a text prompt (text='grape cluster').
    Used on its own it is an automatic model (ByteTrack ids); it can also feed boxes to SAM 2 (DetectThenSAM2)."""
    tracks = True
    def __init__(self, weights, text=None, conf=0.25, imgsz=1280, tracker="bytetrack.yaml"):
        self.weights, self.text, self.conf, self.imgsz, self.tracker = str(weights), text, conf, imgsz, tracker
        self._m = None
    @property
    def model(self):
        if self._m is None:
            ul = _ultralytics(text=bool(self.text))
            print(f"loading {Path(self.weights).name} (downloaded on first use) …")
            with contextlib.redirect_stdout(io.StringIO()):          # hides the download progress bars
                if self.text:
                    names = [self.text] if isinstance(self.text, str) else list(self.text)
                    self._m = ul.YOLOE(self.weights)
                    self._m.set_classes(names, self._m.get_text_pe(names))
                else:
                    self._m = ul.YOLO(self.weights)
        return self._m
    def prepare(self):
        self.model
    def _tracker_file(self):
        """ByteTrack only starts a track above its own score bar (0.25 by default), which would silently
        discard every detection between `conf` and 0.25. Tie the tracker's bars to `conf` instead."""
        import yaml, ultralytics
        base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / self.tracker
        cfg = yaml.safe_load((base if base.exists() else Path(self.tracker)).read_text())
        cfg.update(track_high_thresh=float(self.conf), new_track_thresh=float(self.conf),
                   track_low_thresh=float(min(cfg.get("track_low_thresh", 0.1), self.conf / 2)))
        f = CKPT_DIR / f"{Path(self.tracker).stem}_conf{self.conf:g}.yaml"
        f.write_text(yaml.safe_dump(cfg))
        return str(f)
    def _unpack(self, r, hw):
        n = 0 if r.boxes is None else len(r.boxes)
        boxes = r.boxes.xyxy.cpu().numpy() if n else np.zeros((0, 4), np.float32)
        scores = r.boxes.conf.cpu().numpy() if n else np.zeros(0, np.float32)
        ids = r.boxes.id.int().cpu().numpy() if n and r.boxes.id is not None else None
        if n and r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5
            if masks.shape[1:] != tuple(hw):
                masks = np.stack([cv2.resize(m.astype(np.uint8), hw[::-1], interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
        else:                                       # box-only weights: fall back to filled boxes
            masks = np.zeros((n, *hw), bool)
            for i, (x0, y0, x1, y1) in enumerate(boxes.round().astype(int)):
                masks[i, max(y0, 0):y1, max(x0, 0):x1] = True
        return boxes, scores, masks, ids
    def detect(self, img):
        """One RGB frame → (boxes [N,4] x0,y0,x1,y1, scores [N], masks [N,H,W])."""
        r = self.model.predict(img[..., ::-1], conf=self.conf, imgsz=self.imgsz, retina_masks=True, verbose=False)[0]
        return self._unpack(r, img.shape[:2])[:3]
    def predict(self, clip, prompts=None):
        T = len(clip); H, W = clip.hw
        out, remap, tracker = np.zeros((T, H, W), np.uint16), {}, self._tracker_file()
        for t in range(T):
            r = self.model.track(clip.image(t)[..., ::-1], persist=t > 0, conf=self.conf, imgsz=self.imgsz,
                                 tracker=tracker, retina_masks=True, verbose=False)[0]
            boxes, scores, masks, ids = self._unpack(r, (H, W))
            if ids is None: continue
            for i in np.argsort(scores):            # most confident painted last
                out[t][masks[i]] = remap.setdefault(int(ids[i]), len(remap) + 1)
        return out

def _same_box(a, b, iou_thr):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6) >= iou_thr or inter / max(min(area_a, area_b), 1e-6) >= 0.7

class DetectThenSAM2(Model):
    """The automatic SAM 2 pipeline: a detector finds objects on every `every`-th frame, SAM 2.1 turns each
    new box into a mask and follows it. SAM 2 cannot find grapes by itself; it only follows what it is shown."""
    tracks = True
    def __init__(self, detector, size=None, ckpt=None, every=4, match_iou=0.4, max_objects=200, backward=False):
        self.detector, self.size, self.ckpt = detector, size or SAM2_SIZE, ckpt
        self.every, self.match_iou, self.max_objects, self.backward = every, match_iou, max_objects, backward
        self.last_prompts = []
    def prepare(self):
        self.detector.prepare(); load_sam2(self.size, self.ckpt)
    def predict(self, clip, prompts=None):
        sess = Sam2Session(load_sam2(self.size, self.ckpt), clip)
        ids = np.zeros((sess.T, sess.H, sess.W), np.uint16); best = np.zeros(ids.shape, np.float16)
        made = []
        for t in range(0, sess.T, self.every):
            boxes, scores, _ = self.detector.detect(clip.image(t))
            known = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(ids[t]) if s is not None]
            new = []
            for b in boxes[np.argsort(-scores)]:
                if len(made) + len(new) >= self.max_objects: break
                if any(_same_box(b, k, self.match_iou) for k in known): continue
                known.append(tuple(b))
                new.append(dict(obj=len(made) + len(new) + 1, frame=t, box=[float(v) for v in b]))
            if new:
                sess.add(new, ids, best, backward=self.backward); made += new
        self.last_prompts = made
        return ids

class FromFolder(Model):
    """Score predictions written by any other program: <folder>/<clip>/<frame>.png, pixel value = object id."""
    def __init__(self, folder, tracks=False):
        self.folder, self.tracks = Path(folder), tracks
    def predict(self, clip, prompts=None):
        lab = clip.labels(root=self.folder)
        if lab is None:
            raise FileNotFoundError(f"{self.folder / clip.name} needs one PNG per frame, named like the frames")
        return lab

class SAM3Text(Model):
    """SAM 3.1 (or SAM 3) with a text prompt, through Meta's own `sam3` package. Needs that package installed
    and Hugging Face access to the gated weights. NOT run while this pipeline was tested. See Appendix B."""
    tracks = True
    def __init__(self, text=None, version="sam3.1", max_num_objects=64, **builder_args):
        self.text, self._p = text or OBJECT_NAME, None
        self.args = dict(version=version, **builder_args)
        if version == "sam3.1":
            self.args["max_num_objects"] = max_num_objects      # Meta's default of 16 is too few for a vine row
    def prepare(self):
        if self._p is None:
            import sam3.model_builder as mb
            if hasattr(mb, "build_sam3_predictor"):              # March 2026 and later: SAM 3 and SAM 3.1
                self._p = mb.build_sam3_predictor(**self.args)
            else:                                                # original SAM 3 release
                self._p = mb.build_sam3_video_predictor(**{k: v for k, v in self.args.items()
                                                            if k not in ("version", "max_num_objects")})
    def predict(self, clip, prompts=None):
        self.prepare()
        T = len(clip); H, W = clip.hw
        out = np.zeros((T, H, W), np.uint16)
        sid = self._p.handle_request(dict(type="start_session", resource_path=str(clip.dir)))["session_id"]
        try:
            self._p.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=self.text))
            for resp in self._p.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                o = resp["outputs"]
                for k, m in zip(np.asarray(o["out_obj_ids"]).tolist(), np.asarray(o["out_binary_masks"])):
                    m = np.asarray(m, bool)
                    if m.shape != (H, W):
                        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    out[resp["frame_index"]][m] = int(k) + 1
        finally:
            self._p.handle_request(dict(type="close_session", session_id=sid))
        return out

# ---- describing models in plain data (used by the HPCC jobs and the lineup files) ------------------------
def _project_path(p):
    """A path written in a lineup file: absolute, ~/…, or relative to the project folder."""
    p = Path(os.path.expandvars(str(p))).expanduser()
    return p if p.is_absolute() else ROOT / p

def build_model(spec):
    """Make a model from a plain dict, e.g. one entry of a lineup YAML file:
      {type: sam2, size: base_plus, checkpoint: models/x.pt, prompt: box}      SAM 2.1, pretrained or fine-tuned
      {type: ultralytics, weights: yoloe-26m-seg.pt, text: grape cluster}      YOLO / YOLOE (automatic)
      {type: detect_then_sam2, detector: {...}, checkpoint: models/x.pt}       a detector finds, SAM 2.1 follows
      {type: folder, path: predictions/sam3_text, tracks: true}                masks written by any other program
      {type: sam3, text: grape cluster}                                        SAM 3.1 (untested, gated weights)
      {type: python, class: my_models.py:MyModel, args: {...}}                 your own Model subclass"""
    spec = dict(spec); kind = spec.pop("type").lower()
    ckpt = spec.pop("checkpoint", None)
    ckpt = _project_path(ckpt) if ckpt else None
    if kind == "sam2":
        return SAM2Prompted(spec.get("size"), ckpt, spec.get("prompt", "box"))
    if kind in ("ultralytics", "yolo", "yoloe"):
        w = str(spec.pop("weights"))
        w = str(_project_path(w)) if ("/" in w or _project_path(w).exists()) else w      # bare names: Ultralytics fetches them
        return UltralyticsModel(w, **spec)
    if kind == "detect_then_sam2":
        return DetectThenSAM2(build_model(spec.pop("detector")), spec.pop("size", None), ckpt, **spec)
    if kind == "folder":
        return FromFolder(_project_path(spec["path"]), tracks=bool(spec.get("tracks", False)))
    if kind == "sam3":
        return SAM3Text(**spec)
    if kind == "python":
        target, args = spec.pop("class"), spec.pop("args", {}) or {}
        mod_name, cls_name = target.rsplit(":", 1)
        import importlib, importlib.util
        if mod_name.endswith(".py"):
            path = _project_path(mod_name) if not Path(mod_name).expanduser().exists() else Path(mod_name).expanduser()
            spec_ = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec_)
            sys.modules[path.stem] = mod                       # its classes can then subclass Model from here
            mod.__dict__.update({k: v for k, v in globals().items() if k in ("Model", "np", "cv2", "Clip")})
            spec_.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)(**args)
    raise ValueError(f"unknown model type '{kind}' (sam2, ultralytics, detect_then_sam2, folder, sam3, python)")

def load_lineup(path):
    """Read a lineup YAML file → {name: Model}. See configs/lineup.yaml for a commented example.
    A `python` model's file may be given relative to the lineup file itself."""
    import yaml
    path = Path(path).expanduser()
    entries = yaml.safe_load(path.read_text())["models"] or []
    out = {}
    for e in entries:
        e = dict(e)
        if e.pop("skip", False):
            continue
        if e.get("type") == "python" and ".py:" in e.get("class", ""):
            f, cls = e["class"].rsplit(":", 1)
            if not Path(f).expanduser().is_absolute() and (path.parent / f).exists():
                e["class"] = f"{(path.parent / f).resolve()}:{cls}"
        name = e.pop("name")                 # (popped before build_model sees the entry)
        out[name] = build_model(e)
    return out

# ---- run + score ------------------------------------------------------------------------------------
PROMPT_VISIBLE_FRAC = 0.5    # prompt an object on the first frame where half of its largest view is in frame
MIN_OBJECT_AREA = 100        # automatic track: ignore specks smaller than this many pixels (labels and predictions)

def run_model(name, model, clips):
    """Predict every clip and save the result under predictions/<name>/. Returns seconds per frame."""
    root, secs, n = PRED / _safe(name), 0.0, 0
    model.prepare()
    for clip in tqdm(clips, desc=name, leave=False):
        prompts = prompts_from_labels(clip.labels(), model.prompt, PROMPT_VISIBLE_FRAC) if model.needs_prompts else None
        t0 = time.time()
        pred = np.asarray(model.predict(clip, prompts))
        secs += time.time() - t0; n += len(clip)
        assert pred.shape == (len(clip), *clip.hw), f"{name}: expected {(len(clip), *clip.hw)}, got {pred.shape}"
        clip.save_labels(pred, root=root)
    return secs / max(n, 1)

def score_model(name, model, clips):
    """Compare saved predictions with the labels. Returns (summary row, detail table)."""
    root, obj_rows, frame_rows, uniq = PRED / _safe(name), [], [], [0, 0]
    for clip in clips:
        gt, pred = clip.labels(), clip.labels(root=root)
        if model.needs_prompts:
            prompts = prompts_from_labels(gt, "box", PROMPT_VISIBLE_FRAC)        # same frames for every prompt type
            obj_rows += [dict(model=name, clip=clip.name, **r) for r in score_prompted(gt, pred, prompts)]
        else:
            frame_rows += [dict(model=name, clip=clip.name, **r) for r in score_automatic(gt, pred, 0.5, MIN_OBJECT_AREA)]
            for i, arr in enumerate((gt, pred)):
                k, c = np.unique(arr, return_counts=True)
                uniq[i] += int(((k != 0) & (c >= MIN_OBJECT_AREA)).sum())
    if not obj_rows and not frame_rows:
        raise RuntimeError("nothing to score: these clips have no labelled objects")
    if model.needs_prompts:
        d = pd.DataFrame(obj_rows)
        J, F = d["J"].mean(), d["F"].mean()
        row = {"model": name, "kind": "prompted", "prompt": model.prompt, "objects": len(d),
               "J&F": (J + F) / 2, "J": J, "F": F, "J&F (DAVIS)": (d["J_all"].mean() + d["F_all"].mean()) / 2,
               "IoU on prompt frame": np.nan if model.prompt == "mask" else d["iou_prompt_frame"].mean()}
    else:
        d = pd.DataFrame(frame_rows)
        tp, fp, fn = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
        diff = d["n_pred"] - d["n_gt"]
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        row = {"model": name, "kind": "automatic", "precision": 100 * P, "recall": 100 * R,
               "F1": 100 * 2 * P * R / max(P + R, 1e-9), "mask IoU (matched)": 100 * d["iou_sum"].sum() / max(tp, 1),
               "count error / frame": diff.abs().mean(), "count bias / frame": diff.mean(),
               "objects / frame (truth)": d["n_gt"].mean(),
               "tracks found / true": f"{uniq[1]} / {uniq[0]}" if model.tracks else "n/a"}
    return row, d

def _data_signature(clips):
    """Short fingerprint of the clips and their labels: rows scored on different data are never shown together."""
    import hashlib
    h = hashlib.md5()
    for c in sorted(clips, key=lambda c: c.name):
        h.update(c.name.encode())
        for f in sorted((ANN / c.name).glob("*.png")): h.update(f.read_bytes())
    return h.hexdigest()[:8]

def benchmark(models, split="val", clips=None):
    """Run and score every model in `models` (a dict name → Model) on the labelled clips of `split`.
    Rows add up across calls, so the pretrained models scored earlier sit next to the fine-tuned ones."""
    clips = clips or get_clips(split, labelled=True)
    if not clips:
        raise RuntimeError(f"No labelled '{split}' clips yet. Label some in Step 3 and mark them done "
                           "(Step 3c shows what is missing).")
    sig, rows, details = _data_signature(clips), [], {}
    for name, model in models.items():
        try:
            spf = run_model(name, model, clips)
            row, details[name] = score_model(name, model, clips)
            rows.append({**row, "sec / frame": spf, "data": sig})
        except Exception as e:                       # one broken model should not sink the whole table
            print(f"✗ {name}: {type(e).__name__}: {str(e)[:300]}")
    RESULTS.mkdir(exist_ok=True)
    f, fd = RESULTS / f"leaderboard_{split}.csv", RESULTS / f"leaderboard_{split}_details.csv"
    old = pd.read_csv(f) if f.exists() and f.stat().st_size > 5 else pd.DataFrame()
    if len(old):
        stale = int((old["data"] != sig).sum())
        if stale: print(f"({stale} earlier row(s) were scored on a different set of clips or labels and were dropped)")
        old = old[(old["data"] == sig) & ~old["model"].isin([r["model"] for r in rows])]
    board = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
    if len(board): board.to_csv(f, index=False)
    if details:
        new = pd.concat(details.values()).assign(data=sig)
        if fd.exists():
            od = pd.read_csv(fd); new = pd.concat([od[(od["data"] == sig) & ~od["model"].isin(list(details))], new])
        new.to_csv(fd, index=False)
    print(f"scored on {len(clips)} {split} clip(s), {sum(len(c) for c in clips)} frames · table saved to {f}")
    show_board(board)
    own = [c.name for c in clips if _load_prompts(c.name)]
    if own and len(board) and (board["kind"] == "prompted").any():
        print(f"\n⚠️  {len(own)} of these {len(clips)} clips were labelled with SAM 2.1's help (labelling screen or box imports), so SAM 2.1 is partly being "
              "graded against its own drawing. Its 'box' rows read high for that reason alone. Trust differences on the "
              "click rows and in the automatic table more, and see 'Keeping the score honest' under Step 3.")
    return board

def clear_leaderboard(split="val"):
    """Start the table over (the saved predictions stay)."""
    for f in RESULTS.glob(f"leaderboard_{split}*.csv"): f.unlink()

def show_board(board):
    for kind, cols, key in (("prompted", ["model", "prompt", "J&F", "J", "F", "J&F (DAVIS)", "IoU on prompt frame", "objects", "sec / frame"], "J&F"),
                            ("automatic", ["model", "F1", "precision", "recall", "mask IoU (matched)", "count error / frame",
                                           "count bias / frame", "objects / frame (truth)", "tracks found / true", "sec / frame"], "F1")):
        part = board[board["kind"] == kind] if len(board) else board
        if len(part):
            print(f"\n{kind.upper()} models" + (" — shown each object once, must follow it" if kind == "prompted"
                                                   else " — must find every object themselves"))
            display(part[cols].sort_values(key, ascending=False).round(2).reset_index(drop=True))
