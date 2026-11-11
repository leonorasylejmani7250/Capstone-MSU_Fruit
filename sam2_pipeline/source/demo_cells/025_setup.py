# Step 1: load the pipeline code (fruit_sam2.py, in this folder) and pick the smallest SAM 2.1, so the demo runs on a laptop.
import os, sys, subprocess
from pathlib import Path

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
import fruit_sam2 as F        # first run: downloads Meta's SAM 2 code (a few MB)

F.SAM2_SIZE = "tiny"          # the smallest SAM 2.1 (39M parameters), so the demo runs on a laptop
if F.DEVICE == "cpu":
    F.SAM2_IMAGE_SIZE = 512   # faster on a CPU, a little less accurate; GPUs use SAM 2's full 1024
F.use_project("../data/demo")  # the repository's shared data folder; the demo keeps to its own subfolder
print("running on", F.GPU_NAME or "CPU", "· project folder:", F.ROOT)
