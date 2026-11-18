#!/bin/bash --login
# Builds an environment of its own for the YOLO26 pipeline on the MSU HPCC. It is optional, and hpcc/setup.sh in
# the repository folder does not run it: the SAM 2.1 environment (~/envs/fruit_sam2) holds Ultralytics and runs
# this folder's scripts. Build this one to change this pipeline's packages without touching SAM 2.1. Once it
# exists, activate.sh and run.sb of this folder and hpcc/count_rows.sh use it in place of the SAM 2.1 environment
# (hpcc/run_all.sh fine-tunes YOLO through the SAM 2.1 pipeline, in the SAM 2.1 environment, either way).
# Run it on a development node, because those have internet access (compute nodes may not):
#     ssh dev-amd24-h200
#     cd ~/Capstone-MSU_Fruit/yolo26_pipeline
#     bash hpcc/setup_env.sh
# Safe to run again: it only adds what is missing. Takes about 10 minutes the first time.
#
# WHAT IT MAKES
#   ~/envs/fruit_yolo26                  the conda environment: Python 3.12, PyTorch, the packages in requirements.txt
#   yolo26n-seg.pt, yolo26s-seg.pt       Ultralytics' base weights, in this folder (git ignores *.pt files)
#   ~/.config/Ultralytics/               Ultralytics' settings file, and the font it labels its plots with
# It edits no login file, submits no job and trains nothing. The modules and the environment it loads are gone
# again when it ends.
#
# TO CHANGE IT
#   a package            edit requirements.txt (it says what each line is for), run this script again
#   the Python version   edit python=3.12 below, delete the folder ~/envs/fruit_yolo26, run this script again
#   the PyTorch build    after `source hpcc/activate.sh`: python -m pip uninstall -y torch torchvision; then edit
#                        the --index-url line below and run this script again
#   where it lives       the default path below is also in activate.sh and run.sb of this folder and in
#                        hpcc/count_rows.sh of the repository folder: change all of them
#   back to the SAM 2.1 environment: delete the folder ~/envs/fruit_yolo26
set -eo pipefail          # stop at the first command that fails (no -u: conda's activate scripts read unset variables)
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # this pipeline's folder, wherever this is run from
YOLO26_ENV_DIR="${YOLO26_ENV_DIR:-$HOME/envs/fruit_yolo26}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts

# The HPCC offers its software as "modules". The first line unloads the ones every login starts with: Miniforge3
# does not load next to them, and their Python 3.11 packages would get mixed into ours. The second makes conda
# available, the third makes `conda activate` work inside a script. All three hold only until this script ends.
module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
# 1. The conda environment: a folder with its own Python 3.12 and pip, created only when it is not there yet
#    (-p: at this path; -y: without asking). `conda activate` then makes `python` and `pip` mean the
#    environment's own for the rest of this script.
if [ ! -x "$YOLO26_ENV_DIR/bin/python" ]; then
    conda create -y -p "$YOLO26_ENV_DIR" python=3.12 pip
fi
conda activate "$YOLO26_ENV_DIR"
python -m pip install --upgrade pip
# 2. PyTorch, first, so that pip keeps this build when Ultralytics asks for PyTorch. cu126 is the
#    CUDA 12.6 build: the newest PyTorch build that still runs on the cluster's V100s as well as its newer GPUs.
#    No version is named, so pip takes the newest PyTorch that index has on the day the environment is built.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 3. The other packages: Ultralytics and the tracker's helper.
python -m pip install -r "$CODE_DIR/requirements.txt"
# 4. Fetch the base weights and the plotting font now, so a job on a compute node never needs the internet.
#    Ultralytics downloads a weights file it is asked for by name into the current folder, hence the cd.
#    n is the size train.py starts from, s the bigger one of the README's second training command; add
#    another name to the line to fetch it as well.
cd "$CODE_DIR"
python -c "from ultralytics import YOLO; YOLO('yolo26n-seg.pt'); YOLO('yolo26s-seg.pt'); from ultralytics.utils.checks import check_font; check_font('Arial.ttf'); print('base weights (n, s) and font ready')"
# 5. Check: prints the versions that were installed and whether this node has a GPU that PyTorch can use.
python -c "import torch, ultralytics; print('ultralytics', ultralytics.__version__, '· torch', torch.__version__, '· CUDA', torch.cuda.is_available())"
echo "Next: sbatch hpcc/run.sb train.py --epochs 50 --imgsz 512 --batch 16"
