#!/bin/bash --login
# Build an environment of its own for the YOLO26 pipeline on the MSU HPCC. This is
# optional, and hpcc/setup.sh does not run it: the SAM 2.1 environment holds Ultralytics
# and runs this folder's scripts. Build this one to change this pipeline's packages
# without touching SAM 2.1. Run it on a development node (those have internet access):
#
#    ssh dev-amd24-h200
#    cd ~/Capstone-MSU_Fruit/yolo26_pipeline
#    bash hpcc/setup_env.sh
#
# It makes ~/envs/fruit_yolo26, a conda environment with Python 3.12, PyTorch and the
# packages of requirements.txt, and fetches Ultralytics' base weights into this folder.
# It takes about 10 minutes and is safe to run again. Once the environment exists,
# activate.sh and run.sb of this folder and hpcc/count_rows.sh use it; deleting the
# folder goes back to the SAM 2.1 environment.
#
# How to change the environment: hpcc/README.md in the repository folder, "The
# environments".

# Stop at the first command that fails. (No -u: conda's activate scripts read unset
# variables.)
set -eo pipefail

# This pipeline's folder, wherever this script is run from.
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO26_ENV_DIR="${YOLO26_ENV_DIR:-$HOME/envs/fruit_yolo26}"

# Make conda available: unload the modules every login starts with, load Miniforge3, and
# let `conda activate` work inside this script.
module purge
module load Miniforge3
eval "$(conda shell.bash hook)"

# 1. The conda environment, created only when it is not there yet.
if [ ! -x "$YOLO26_ENV_DIR/bin/python" ]; then
    conda create -y -p "$YOLO26_ENV_DIR" python=3.12 pip
fi
conda activate "$YOLO26_ENV_DIR"
python -m pip install --upgrade pip

# 2. PyTorch first, so that pip keeps this build when Ultralytics asks for PyTorch.
# cu126 is the CUDA 12.6 build, the newest that still runs on the cluster's V100s.
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# 3. The packages of requirements.txt.
python -m pip install -r "$CODE_DIR/requirements.txt"

# 4. Ultralytics' base weights (the n and s sizes) and the font for its plots, fetched
# now so that jobs do not need the internet. Ultralytics saves weights into the current
# folder.
cd "$CODE_DIR"
python -c "
from ultralytics import YOLO
YOLO('yolo26n-seg.pt')
YOLO('yolo26s-seg.pt')
from ultralytics.utils.checks import check_font
check_font('Arial.ttf')
print('base weights (n, s) and font ready')
"

# 5. Check: the line it prints ends with whether this node has a GPU that PyTorch can
# use.
python -c "
import torch, ultralytics
print(
    'ultralytics', ultralytics.__version__,
    '· torch', torch.__version__,
    '· CUDA', torch.cuda.is_available(),
)
"
echo "Next: sbatch hpcc/run.sb train.py --epochs 50 --imgsz 512 --batch 16"
