#@title Tools: fine-tuning SAM 2.1 (Meta's trainer) and training a YOLO detector  { display-mode: "form" }
_LADDER = [(8, 3), (6, 3), (4, 3), (4, 2), (4, 1), (2, 1)]      # (frames per sample, objects per sample)

def _auto_train_plan(encoder="auto"):
    if encoder == "yes":  freeze = False
    elif encoder == "no": freeze = True
    else:                 freeze = GPU_GB < 38
    start = 0 if GPU_GB >= 20 else 3 if DEVICE == "cuda" else 5
    return freeze, start

def write_train_config(run, train_clips, size, steps, lr, freeze, frames, objects, init_ckpt=None,
                       steps_per_epoch=100, data_root=None, workers=2 if DEVICE == "cuda" else 0, amp_dtype="bfloat16"):
    """Start from Meta's official fine-tuning recipe and change only what a small single-GPU run needs."""
    from omegaconf import OmegaConf
    cfg_dir = SAM2_REPO / "sam2" / "configs" / "sam2.1_training"
    cfg = OmegaConf.load(cfg_dir / "sam2.1_hiera_b+_MOSE_finetune.yaml")
    inf = OmegaConf.load(SAM2_REPO / "sam2" / SAM2_MODELS[size]["cfg"])
    run_dir = RUNS / run; run_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root or DATA)

    # backbone of the chosen size (the recipe file is written for base_plus)
    enc = cfg.trainer.model.image_encoder
    enc.trunk = OmegaConf.merge(inf.model.image_encoder.trunk, {"drop_path_rate": 0.0 if freeze else 0.1})
    enc.neck.backbone_channel_list = inf.model.image_encoder.neck.backbone_channel_list
    cfg.trainer.model.freeze_image_encoder = bool(freeze)
    # The recipe may hand the model ground-truth masks on up to 2 frames; with only 2 frames per sample that
    # would leave nothing to predict (and nothing to learn from), so keep at least one frame to track.
    cfg.trainer.model.num_init_cond_frames_for_train = max(1, min(2, frames - 1))

    # data: clips with enough frames train as video, single frames (imported images) train as images
    video = [c for c in train_clips if len(c) >= max(frames, 2)]
    image = [c for c in train_clips if len(c) == 1]
    skipped = [c.name for c in train_clips if c not in video and c not in image]
    template = cfg.trainer.data.train.datasets[0]
    entries, n_samples = [], 0
    for kind, clips, n_frames in (("video", video, frames), ("image", image, 1)):
        if not clips: continue
        (run_dir / f"train_{kind}.txt").write_text("\n".join(c.name for c in clips) + "\n")
        mult = max(1, round(min(steps_per_epoch, steps) / (len(video) + len(image))))   # ≈ steps_per_epoch samples per epoch
        e = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
        d = e.dataset.datasets[0]
        d.video_dataset.img_folder = str(data_root / "JPEGImages")
        d.video_dataset.gt_folder = str(data_root / "Annotations")
        d.video_dataset.file_list_txt = str(run_dir / f"train_{kind}.txt")
        d.sampler.num_frames, d.sampler.max_num_objects, d.multiplier = n_frames, objects, mult
        entries.append(e); n_samples += len(clips) * mult
    assert entries, "no usable training clips"
    cfg.trainer.data.train.datasets = entries
    cfg.trainer.data.train.batch_sizes = [1] * len(entries)
    if len(entries) > 1:
        cfg.trainer.data.train.dataset_prob = [0.5, 0.5]
    cfg.trainer.data.train.num_workers = workers

    epochs = max(1, math.ceil(steps / n_samples))
    cfg.scratch.update(dict(resolution=int(SAM2_IMAGE_SIZE), num_frames=frames, max_num_objects=objects,
                            num_epochs=epochs, num_train_workers=workers, base_lr=float(lr),
                            vision_lr=float(lr) * 0.6, train_batch_size=1))
    cfg.dataset.update(dict(img_folder=str(data_root / "JPEGImages"), gt_folder=str(data_root / "Annotations"),
                            file_list_txt=str(run_dir / "train_video.txt")))
    cfg.trainer.accelerator = DEVICE
    cfg.trainer.distributed.backend = "nccl" if DEVICE == "cuda" else "gloo"
    cfg.trainer.optim.amp.enabled = DEVICE == "cuda"
    cfg.trainer.optim.amp.amp_dtype = amp_dtype      # bfloat16 needs an A100/L4/L40S/H100-class GPU; T4 and V100 use float16
    cfg.trainer.logging.log_freq = int(max(1, min(10, steps // 50)))   # short runs log every step, for the loss plot
    cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = str(init_ckpt or sam2_checkpoint(size))
    cfg.launcher.gpus_per_node = 1
    cfg.launcher.experiment_log_dir = str(run_dir)

    name = f"fruit_{_safe(run)}.yaml"
    (cfg_dir / name).write_text("# @package _global_\n\n" + OmegaConf.to_yaml(cfg))
    plan = dict(run=run, size=size, image_size=int(SAM2_IMAGE_SIZE), steps=epochs * n_samples, epochs=epochs,
                steps_per_epoch=n_samples, lr=float(lr), freeze_encoder=bool(freeze), frames=frames, objects=objects,
                video_clips=len(video), image_clips=len(image), skipped=skipped,
                amp_dtype=amp_dtype if DEVICE == "cuda" else "off",
                train_clips=[c.name for c in video + image], sam2_commit=SAM2_COMMIT)
    return f"configs/sam2.1_training/{name}", plan

def _launch(config_name, run_dir, total_steps):
    """Run Meta's trainer as a subprocess; print one short progress line per log tick. Returns (ok, log tail)."""
    env = dict(os.environ, PYTHONPATH=str(SAM2_REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    args = ["training/train.py", "-c", config_name, "--use-cluster", "0", "--num-gpus", "1"]
    cmd = [sys.executable] + args
    if DEVICE != "cuda":       # Meta's trainer resets GPU memory statistics every step; make that a no-op on CPU
        cmd = [sys.executable, "-c", "import sys, runpy, torch; torch.cuda.reset_peak_memory_stats = lambda *a, **k: None; "
               f"sys.argv = {args!r}; runpy.run_path('training/train.py', run_name='__main__')"]
    log = open(run_dir / "console.log", "a")
    proc = subprocess.Popen(cmd, cwd=SAM2_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail, pat = [], re.compile(r"Train Epoch: \[(\d+)\]\[\s*(\d+)/\s*(\d+)\] \| Batch Time: [\d.]+ \(([\d.]+)\).*?"
                               r"Mem \(GB\): [\d.]+ \([\d.]+/([\d.]+)\).*?Losses/train_all_loss: [\d.eE+-]+ \(([\d.eE+-]+)\)")
    shown, every = -10**9, max(10, total_steps // 15)          # about 15 progress lines per run
    for line in proc.stdout:
        log.write(line); tail = (tail + [line])[-60:]
        m = pat.search(line)
        if m:
            ep, it, per = int(m[1]), int(m[2]), int(m[3])
            log.flush()
            step = ep * per + it
            if step - shown < every and step < total_steps - 1:
                continue
            shown = step
            print(f"  step {step:>5}/{total_steps} · average loss this epoch {float(m[6]):.3f} · {float(m[4]):.1f} s/step"
                  + (f" · peak GPU memory {float(m[5]):.0f} GB" if DEVICE == "cuda" else ""), flush=True)
    proc.wait(); log.close()
    return proc.returncode == 0, "".join(tail)

def finetune(run="grapes_v1", size=None, steps=1500, lr=5e-6, encoder="auto", clips=None, init_ckpt=None,
             amp_dtype="auto"):
    """Fine-tune SAM 2.1 with Meta's trainer on the labelled train clips and return the path of the exported
    weights. Re-running with the same name resumes an interrupted run, or reuses the finished weights.
    encoder: "auto" | "no" (freeze the image backbone, fits 16 GB) | "yes" (train it too, ~40 GB+).
    amp_dtype: "auto" (bfloat16 on A100/L4/L40S/H100-class GPUs, float16 on T4/V100) | "bfloat16" | "float16"."""
    size = size or SAM2_SIZE
    clips = clips or get_clips("train", labelled=True)
    if not clips:
        raise RuntimeError("No labelled training clips yet. Label some in Step 3 and mark them done.")
    out = WEIGHTS / f"sam2.1_{size}_{_safe(run)}.pt"
    if out.exists():
        print(f"{out.name} already exists, reusing it. Pick a new run name to train again."); return out
    over = [c.name for c in clips if int(c.labels().max()) > 255]
    if over:
        raise RuntimeError(f"Meta's trainer reads at most 255 objects per clip; these have more: {over[:5]}")
    run_dir = RUNS / run
    data_root = DATA
    if "/drive/" in str(DATA):            # Drive is slow for thousands of small reads: train from a fresh local copy
        data_root = BASE / "train_data_local"
        shutil.rmtree(data_root, ignore_errors=True)
        for sub in ("JPEGImages", "Annotations"):
            for c in clips:
                shutil.copytree(DATA / sub / c.name, data_root / sub / c.name)
    freeze, rung = _auto_train_plan(encoder)
    if amp_dtype == "auto":
        amp_dtype = "bfloat16" if BF16_OK else "float16"
    shortest = min((len(c) for c in clips if len(c) > 1), default=1)
    while rung < len(_LADDER) - 1 and _LADDER[rung][0] > shortest: rung += 1
    while True:
        frames, objects = _LADDER[rung]
        config_name, plan = write_train_config(run, clips, size, steps, lr, freeze, min(frames, max(shortest, 1)),
                                               objects, init_ckpt, data_root=data_root, amp_dtype=amp_dtype)
        old = json.loads((run_dir / "plan.json").read_text()) if (run_dir / "plan.json").exists() else plan
        if old != plan:                    # same name, different settings: do not resume the old attempt
            print("  settings changed since the last attempt under this name, starting it over")
            for sub in ("checkpoints", "logs", "tensorboard"): shutil.rmtree(run_dir / sub, ignore_errors=True)
        (run_dir / "plan.json").write_text(json.dumps(plan, indent=1))
        print(f"training {size}: {plan['steps']} steps on {plan['video_clips']} video + {plan['image_clips']} image clip(s), "
              f"{plan['frames']} frames × {objects} objects per step, encoder {'frozen' if freeze else 'trained'}, lr {lr:g}"
              + (f", {amp_dtype} on {GPU_NAME}" if DEVICE == "cuda" else ", CPU"))
        if plan["skipped"]: print("  skipped (too short for video training):", plan["skipped"])
        t0 = time.time()
        ok, tail = _launch(config_name, run_dir, plan["steps"])
        if ok: break
        if "out of memory" in tail.lower() and rung < len(_LADDER) - 1:
            rung += 1
            shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
            print(f"  GPU ran out of memory; retrying with {_LADDER[rung][0]} frames × {_LADDER[rung][1]} objects")
            continue
        if "attempting to stop training" in tail and amp_dtype == "float16" and DEVICE == "cuda":
            amp_dtype = "bfloat16"            # float16 overflowed: slower but steadier on this GPU
            shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
            print("  the loss became infinite in float16; retrying in bfloat16 (slower on this GPU)")
            continue
        print(tail[-3000:]); raise RuntimeError(f"training failed, full log: {run_dir / 'console.log'}")
    sd = torch.load(run_dir / "checkpoints" / "checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    # weights only, so it loads like any SAM 2.1 checkpoint; size and resolution travel with the file
    torch.save({"model": sd, "size": size, "image_size": int(SAM2_IMAGE_SIZE)}, out)
    plan["minutes"] = round((time.time() - t0) / 60, 1)
    out.with_suffix(".json").write_text(json.dumps(plan, indent=1))  # the full recipe behind these weights
    print(f"done in {plan['minutes']} min → {out}")
    return out

def plot_training(run="grapes_v1"):
    """Training loss over the run, read from the trainer's log (one reading every 10 steps). Single readings
    jump around because each step sees one clip; the line is their running mean. Full detail: TensorBoard logs
    in runs/<name>/tensorboard."""
    f = RUNS / run / "console.log"
    if not f.exists():
        print("no training log for this run in", f.parent); return
    pts = re.findall(r"Train Epoch: \[(\d+)\]\[\s*(\d+)/\s*(\d+)\].*?Losses/train_all_loss: ([\d.eE+-]+)", f.read_text())
    last = {}
    for ep, it, per, loss in pts: last[int(ep) * int(per) + int(it)] = float(loss)     # a resumed run logs steps twice
    if len(last) < 3:
        print(f"only {len(last)} loss reading(s): too short a run to draw a curve"); return
    x = np.array(sorted(last)); y = np.array([last[k] for k in x])
    k = max(1, min(10, len(y) // 3)); smooth = np.convolve(np.pad(y, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")
    plt.figure(figsize=(7, 2.8)); plt.plot(x, y, ".", alpha=0.35, label="one step"); plt.plot(x, smooth, label=f"mean of last {k} readings")
    plt.xlabel("training step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    if _in_notebook():
        plt.show()
    else:                                  # command line / batch job: keep the picture next to the run
        plt.savefig(f.parent / "loss.png", dpi=120); plt.close(); print("loss curve:", f.parent / "loss.png")

def train_detector(base="yolo26n-seg.pt", epochs=100, imgsz=1280, name="yolo_detector", every=2):
    """Train a YOLO segmentation model on the labelled clips (same train / val split). Returns the path of its
    best weights. `every`: use every n-th frame of each clip (neighbouring frames are near-copies)."""
    ul = _ultralytics()
    best = RUNS / name / "weights" / "best.pt"
    if best.exists():
        print(f"{best} already exists, reusing it. Pick a new name to train again."); return best
    data = export_yolo_seg(every=every)
    model = ul.YOLO(base)
    model.train(data=str(data), epochs=int(epochs), imgsz=int(imgsz), project=str(RUNS),
                name=name, exist_ok=True, device=0 if DEVICE == "cuda" else "cpu", workers=2, plots=False)
    return Path(model.trainer.best)
