#@title B · Install and set up (about a minute)  { display-mode: "form" }
import os, sys, subprocess, json, time, math, random, shutil, re, glob, io, base64, warnings, functools
from pathlib import Path
from dataclasses import dataclass, field

IN_COLAB = "google.colab" in sys.modules
# Work folder for the SAM 2 code, pretrained weights and temporary files (never your labels):
# $FRUIT_BASE if set, /content in Colab, otherwise ~/.fruit_sam2 (what hpcc/setup_env.sh fills in).
BASE = Path(os.environ.get("FRUIT_BASE") or ("/content" if IN_COLAB else Path.home() / ".fruit_sam2")).expanduser()
BASE.mkdir(parents=True, exist_ok=True)

def sh(cmd, check=True, quiet=True):
    """Run a command (a list of arguments, so paths with spaces are safe); show its output only when it fails."""
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True)
    if r.returncode != 0 and check:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    if not quiet:
        print(r.stdout)
    return r

def _missing(*mods):
    import importlib.util
    return [m for m in mods if importlib.util.find_spec(m) is None]

# --- SAM 2 code, pinned to the exact commit this pipeline was tested with -------------------
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"   # facebookresearch/sam2, 2024-12-15 (latest)
SAM2_REPO = Path(os.environ.get("FRUIT_SAM2_REPO") or BASE / "sam2_repo")
if not (SAM2_REPO / "sam2" / "build_sam.py").exists():
    shutil.rmtree(SAM2_REPO, ignore_errors=True)          # a half-finished earlier download starts over
    SAM2_REPO.mkdir(parents=True, exist_ok=True)
    sh(["git", "-C", SAM2_REPO, "init", "-q"])
    sh(["git", "-C", SAM2_REPO, "remote", "add", "origin", "https://github.com/facebookresearch/sam2.git"])
    sh(["git", "-C", SAM2_REPO, "fetch", "-q", "--depth", "1", "origin", SAM2_COMMIT])
    sh(["git", "-C", SAM2_REPO, "checkout", "-q", "FETCH_HEAD"])
if str(SAM2_REPO) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO))        # makes `sam2` and `training` importable, no pip build

# --- Python packages (only what is missing; Colab already has most) --------------------------
_pip = {"hydra": "hydra-core>=1.3.2", "iopath": "iopath>=0.1.10", "fvcore": "fvcore",
        "tensordict": "tensordict>=0.6.0", "submitit": "submitit", "pycocotools": "pycocotools",
        "tensorboard": "tensorboard", "scipy": "scipy", "yaml": "pyyaml",
        "pandas": "pandas", "matplotlib": "matplotlib", "cv2": "opencv-python-headless",
        "anywidget": "anywidget", "ipywidgets": "ipywidgets", "tqdm": "tqdm"}
_need = [_pip[m] for m in _missing(*_pip)]
if _need:
    print("installing:", ", ".join(_need))
    sh([sys.executable, "-m", "pip", "install", "-q", *_need])
    import importlib; importlib.invalidate_caches()

import numpy as np, pandas as pd, cv2, torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from IPython.display import display, HTML, Video, clear_output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    _p = torch.cuda.get_device_properties(0)
    GPU_GB, GPU_NAME = _p.total_memory / 2**30, _p.name
    BF16_OK = _p.major >= 8          # A100 / L4 / L40S / H100 / H200. T4 and V100 train in float16 instead.
    if _p.major >= 8:                # same speed-up the official notebooks use
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    GPU_GB, GPU_NAME, BF16_OK = 0.0, "", False

def _in_notebook():
    try:
        return get_ipython().__class__.__name__ in ("ZMQInteractiveShell", "Shell")   # Jupyter / Colab
    except NameError:
        return False

# --- Project folder ------------------------------------------------------------------------------
def _safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")

def use_project(root):
    """Point every tool at one project folder: videos/, dataset/ (frames + labels), predictions/,
    models/, results/, runs/. The notebook and the HPCC command line both go through here."""
    global ROOT, VIDEOS, DATA, IMG, ANN, PRED, RUNS, WEIGHTS, RESULTS, _META, PROMPTS
    ROOT = Path(root).expanduser().resolve()
    VIDEOS, DATA      = ROOT / "videos", ROOT / "dataset"
    IMG, ANN          = DATA / "JPEGImages", DATA / "Annotations"
    PRED, RUNS        = ROOT / "predictions", ROOT / "runs"
    WEIGHTS, RESULTS  = ROOT / "models", ROOT / "results"
    _META, PROMPTS    = DATA / "clips.json", DATA / "prompts"
    for _d in (VIDEOS, IMG, ANN, PRED, RUNS, WEIGHTS, RESULTS):
        _d.mkdir(parents=True, exist_ok=True)
    return ROOT

CKPT_DIR = Path(os.environ.get("FRUIT_CHECKPOINTS") or BASE / "checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

if not globals().get("_DEFER_PROJECT"):           # the command-line module picks its folder later
    _project = _safe(PROJECT_NAME) or "fruit_sam2"
    _parent = Path(os.environ.get("FRUIT_PROJECTS") or ("/content" if IN_COLAB else Path.home()))
    if DEMO:        # the demo always gets its own folder, never your real project
        _root = (Path(PROJECT_DIR).expanduser().parent if PROJECT_DIR else _parent) / f"{_project}_demo"
    elif PROJECT_DIR:
        _root = PROJECT_DIR
    elif IN_COLAB and SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        _root = Path("/content/drive/MyDrive") / _project
    else:
        _root = _parent / _project
    use_project(_root)
    print(f"torch {torch.__version__} · " + (f"GPU {GPU_NAME} ({GPU_GB:.0f} GB)" if DEVICE == "cuda" else "CPU")
          + f" · SAM 2 code @ {SAM2_COMMIT[:7]}")
    if DEVICE != "cuda":
        print("⚠️  No GPU. Everything still runs, but slowly. In Colab: Runtime ▸ Change runtime type ▸ T4 GPU.")
    print(("🧪 DEMO MODE: public sample grapes, not your videos.  " if DEMO else "") + f"Project folder: {ROOT}")
