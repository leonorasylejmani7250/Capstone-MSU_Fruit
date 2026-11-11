# Step 1: load the pipeline code (fruit_sam2.py, in this folder) and point it at a scratch project folder.
import os, sys, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# Opened in Google Colab straight from GitHub? Then fetch the repository first and work inside its sam2_pipeline folder.
BRANCH = "main"            # the branch of the team repository to fetch
if not Path("fruit_sam2.py").exists() and "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH,
                    "https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git", "/content/Capstone-MSU_Fruit"], check=True)
    os.chdir("/content/Capstone-MSU_Fruit/sam2_pipeline")
if not Path("fruit_sam2.py").exists() and Path("sam2_pipeline/fruit_sam2.py").exists():
    os.chdir("sam2_pipeline")      # Jupyter was started from the repository root instead of sam2_pipeline/
assert Path("fruit_sam2.py").exists(), "Start Jupyter from the sam2_pipeline folder (see INSTALL.md)."

sys.path.insert(0, str(Path.cwd()))
import fruit_sam2 as F          # first run: fetches Meta's SAM 2 code and installs the packages it needs

F.SAM2_SIZE = "tiny"            # tiny | small | base_plus | large (base_plus is the pipeline's default; needs a GPU to be quick)
if F.DEVICE == "cpu":
    F.SAM2_IMAGE_SIZE = 512     # faster on a CPU, a little less accurate; GPUs use SAM 2's full 1024
F.use_project("../data/count_test")                         # the repository's shared data folder; this test keeps to its own subfolder

IMAGES = Path("../data/wgisd_yolo/images/test").resolve()  # the 27 WGISD test photos (never trained on)
assert IMAGES.is_dir(), f"missing {IMAGES}: the WGISD photos should be in data/wgisd_yolo (see data/README.md)"
print("running on", F.GPU_NAME or "CPU", "· photos:", IMAGES, "· results:", F.ROOT)
