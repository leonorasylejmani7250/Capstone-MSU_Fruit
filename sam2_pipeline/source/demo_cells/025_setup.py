# Step 1: load the pipeline code (sam2_pipeline/fruit_sam2.py) and pick the smallest SAM 2.1, so the demo runs on a laptop.
import os, sys, subprocess
from pathlib import Path

# Opened in Google Colab straight from GitHub? Then fetch the rest of the repository next to it first.
BRANCH = "main"            # the branch of the team repository that holds sam2_pipeline/
if not Path("sam2_pipeline/fruit_sam2.py").exists() and "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH,
                    "https://github.com/moehocbaidi/Capstone-MSU_Fruit.git", "/content/Capstone-MSU_Fruit"], check=True)
    os.chdir("/content/Capstone-MSU_Fruit")
assert Path("sam2_pipeline/fruit_sam2.py").exists(), "Start Jupyter from the repository folder (see INSTALL.md)."

sys.path.insert(0, str(Path("sam2_pipeline").resolve()))
import fruit_sam2 as F        # first run: downloads Meta's SAM 2 code (a few MB)

F.SAM2_SIZE = "tiny"          # the smallest SAM 2.1 (39M parameters), so the demo runs on a laptop
if F.DEVICE == "cpu":
    F.SAM2_IMAGE_SIZE = 512   # faster on a CPU, a little less accurate; GPUs use SAM 2's full 1024
F.use_project("data/demo_project")
print("running on", F.GPU_NAME or "CPU", "· project folder:", F.ROOT)
