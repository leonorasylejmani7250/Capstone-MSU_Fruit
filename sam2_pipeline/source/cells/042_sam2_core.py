#@title Tools: SAM 2.1 (load a model, turn hints into tracked masks)  { display-mode: "form" }
import contextlib, gc
import sam2.sam2_video_predictor as _svp, sam2.utils.misc as _misc
_svp.tqdm = _misc.tqdm = lambda it, **kw: it          # silence SAM 2's per-clip progress bars

SAM2_MODELS = {   # the four SAM 2.1 sizes (released 2024-09-29; the newest checkpoints in the SAM 2 line)
    "tiny":      dict(cfg="configs/sam2.1/sam2.1_hiera_t.yaml",  file="sam2.1_hiera_tiny.pt",      mirror="sam2.1_t.pt"),
    "small":     dict(cfg="configs/sam2.1/sam2.1_hiera_s.yaml",  file="sam2.1_hiera_small.pt",     mirror="sam2.1_s.pt"),
    "base_plus": dict(cfg="configs/sam2.1/sam2.1_hiera_b+.yaml", file="sam2.1_hiera_base_plus.pt", mirror="sam2.1_b.pt"),
    "large":     dict(cfg="configs/sam2.1/sam2.1_hiera_l.yaml",  file="sam2.1_hiera_large.pt",     mirror="sam2.1_l.pt"),
}
SAM2_IMAGE_SIZE = 1024   # advanced: 768 or 512 run faster and lighter but a little less accurately (training and inference alike)
_OFFICIAL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
_MIRROR = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

def sam2_checkpoint(size):
    """Path to the pretrained SAM 2.1 weights for `size`, downloading them once."""
    info = SAM2_MODELS[size]; dst = CKPT_DIR / info["file"]
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".part")
    print(f"downloading SAM 2.1 {size} weights (once per session) …")
    try:
        torch.hub.download_url_to_file(_OFFICIAL + info["file"], str(tmp), progress=False)
    except Exception as e:      # Meta's server unreachable: fall back to the copy Ultralytics hosts on GitHub
        print(f"official download failed ({type(e).__name__}); using the GitHub mirror (half-precision copy of the same weights)")
        torch.hub.download_url_to_file(_MIRROR + info["mirror"], str(tmp), progress=False)
        sd = torch.load(tmp, map_location="cpu", weights_only=True)["model"]
        torch.save({"model": {k: v.float() for k, v in sd.items()}}, tmp)
    tmp.replace(dst)
    return dst

_SAM2_OVERRIDES = [   # same inference settings Meta's builder applies, minus hole filling (needs a compiled CUDA op)
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    "++model.binarize_mask_from_pts_for_mem_enc=true",
]
_models, _info = {}, {}

def checkpoint_info(ckpt):
    """Size and resolution a fine-tuned checkpoint was trained at (written next to it by finetune())."""
    if ckpt is None:
        return {}
    key = (str(ckpt), Path(ckpt).stat().st_mtime_ns)
    if key not in _info:
        side = Path(ckpt).with_suffix(".json")
        if side.exists():
            _info[key] = json.loads(side.read_text())
        else:                                   # the same two facts are stored inside the .pt as a fallback
            extra = torch.load(ckpt, map_location="cpu", weights_only=True)
            _info[key] = {k: extra[k] for k in ("size", "image_size") if k in extra}
    return _info[key]

def load_sam2(size=None, ckpt=None):
    """A SAM 2.1 video predictor. ckpt=None → Meta's pretrained weights for `size`; or the path of a fine-tuned
    .pt file (its size and resolution are read from the file, so `size` can be left out)."""
    info = checkpoint_info(ckpt)
    size = info.get("size") or size or SAM2_SIZE
    side = int(info.get("image_size") or SAM2_IMAGE_SIZE)
    key = (size, str(ckpt), side)
    if key not in _models:
        from sam2.build_sam import build_sam2_video_predictor
        _models[key] = build_sam2_video_predictor(
            SAM2_MODELS[size]["cfg"], str(ckpt or sam2_checkpoint(size)), device=DEVICE, apply_postprocessing=False,
            hydra_overrides_extra=_SAM2_OVERRIDES + [f"++model.image_size={side}"])
    return _models[key]

def free_models():
    _models.clear(); gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

def _amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()

class Sam2Session:
    """One clip loaded into SAM 2. `add()` tracks a set of objects and paints them into id maps.

    A prompt is a dict: {"obj": id, "frame": t} plus "box": [x0, y0, x1, y1] and/or "points": [[x, y], ..] with
    "labels": [1 = belongs, 0 = does not, ..], or "mask": bool[H, W]. An object may have prompts on several
    frames; each prompted frame is segmented from its own prompt and the frames in between are tracked."""

    def __init__(self, predictor, clip):
        self.p, self.clip, self.T = predictor, clip, len(clip)
        self.H, self.W = clip.hw
        with torch.inference_mode(), _amp():
            self.state = predictor.init_state(video_path=str(clip.dir))

    def _apply(self, pr):
        if pr.get("mask") is not None:
            self.p.add_new_mask(self.state, int(pr["frame"]), int(pr["obj"]), np.asarray(pr["mask"], bool))
            return
        pts = np.asarray(pr["points"], np.float32).reshape(-1, 2) if pr.get("points") else None
        self.p.add_new_points_or_box(
            self.state, int(pr["frame"]), int(pr["obj"]), points=pts,
            labels=np.asarray(pr["labels"], np.int32) if pts is not None else None,
            box=np.asarray(pr["box"], np.float32) if pr.get("box") is not None else None)

    @torch.inference_mode()
    def add(self, prompts, ids=None, best=None, backward=True):
        """Track every object in `prompts` and paint it into `ids` (the highest positive score owns a pixel).
        Objects whose first prompt is on the same frame are tracked together, each start frame in its own
        pass, the way Meta's VOS script handles objects that appear later. backward=False leaves the frames
        before an object's first prompt empty (the scoring protocol); backward=True fills them in (labelling)."""
        if ids is None: ids = np.zeros((self.T, self.H, self.W), np.uint16)
        if best is None: best = np.zeros((self.T, self.H, self.W), np.float16)
        per_obj = {}
        for pr in prompts: per_obj.setdefault(int(pr["obj"]), []).append(pr)
        groups = {}
        for o, ps in per_obj.items(): groups.setdefault(min(int(p["frame"]) for p in ps), []).append(o)
        with _amp():
            for f0 in sorted(groups):
                self.p.reset_state(self.state)
                for o in groups[f0]:
                    for pr in per_obj[o]: self._apply(pr)
                passes = [dict(start_frame_idx=f0)] + ([dict(start_frame_idx=f0, reverse=True)] if backward and f0 > 0 else [])
                for kw in passes:
                    for t, obj_ids, logits in self.p.propagate_in_video(self.state, **kw):
                        if kw.get("reverse") and t == f0:
                            continue
                        mx, arg = logits[:, 0].float().max(0)
                        mx, arg = mx.cpu().numpy(), arg.cpu().numpy()
                        win = mx > best[t]
                        ids[t][win] = np.asarray(obj_ids, np.uint16)[arg[win]]
                        best[t][win] = np.minimum(mx[win], 1000.0)
        return ids

def sam2_track(predictor, clip, prompts, backward=True):
    """Prompts → [T, H, W] array of object ids for the whole clip."""
    return Sam2Session(predictor, clip).add(prompts, backward=backward)

def prompts_from_labels(gt, kind="box", visible_frac=0.5, min_area=64):
    """Simulate the hint a person (or a detector) would give: one prompt per labelled object, on the first
    frame where at least `visible_frac` of its largest view is in frame. kind: 'box' | 'point' | 'mask'."""
    T = gt.shape[0]; top = int(gt.max())
    area = np.stack([np.bincount(gt[t].ravel(), minlength=top + 1) for t in range(T)])
    out = []
    for k in range(1, top + 1):
        a = area[:, k]
        if a.max() == 0: continue
        ok = a >= max(min_area, visible_frac * a.max())
        f = int(np.argmax(ok)) if ok.any() else int(a.argmax())
        m = gt[f] == k
        pr = dict(obj=k, frame=f)
        if kind == "mask":
            pr["mask"] = m
        elif kind == "box":
            ys, xs = np.nonzero(m)
            pr["box"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        elif kind == "point":      # the pixel deepest inside the object, as in SAM's own click evaluation
            d = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
            y, x = np.unravel_index(int(d.argmax()), d.shape)
            pr["points"], pr["labels"] = [[float(x), float(y)]], [1]
        else:
            raise ValueError(kind)
        out.append(pr)
    return out

