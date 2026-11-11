#!/bin/bash --login
# One-time setup of the YOLO11 environment on the MSU HPCC. Run it on a development node, because those have
# internet access (compute nodes may not):
#     ssh dev-amd20-v100
#     cd ~/Capstone-MSU_Fruit/yolo11_pipeline
#     bash hpcc/setup_env.sh
# Safe to run again: it only adds what is missing. Takes about 10 minutes the first time.
set -eo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO11_ENV_DIR="${YOLO11_ENV_DIR:-$HOME/envs/fruit_yolo11}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
if [ ! -x "$YOLO11_ENV_DIR/bin/python" ]; then
    conda create -y -p "$YOLO11_ENV_DIR" python=3.12 pip
fi
conda activate "$YOLO11_ENV_DIR"
python -m pip install --upgrade pip
# CUDA 12.6 build: the newest PyTorch build that still runs on the cluster's V100s as well as its newer GPUs.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$CODE_DIR/requirements.txt"
# Fetch the base weights and the plotting font now, so a job on a compute node never needs the internet.
cd "$CODE_DIR"
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt'); YOLO('yolo11s-seg.pt'); from ultralytics.utils.checks import check_font; check_font('Arial.ttf'); print('base weights (n, s) and font ready')"
python -c "import torch, ultralytics; print('ultralytics', ultralytics.__version__, '· torch', torch.__version__, '· CUDA', torch.cuda.is_available())"
echo "Next: sbatch hpcc/run.sb train.py --epochs 50 --imgsz 512 --batch 16"
