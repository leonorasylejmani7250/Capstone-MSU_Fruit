#!/bin/bash --login
# One-time setup on the MSU HPCC: conda environment, SAM 2 code, pretrained weights, Jupyter kernel.
#
# Run it on a development node, because those have internet access (compute nodes may not):
#     ssh dev-amd20-v100            # or dev-amd24 / dev-amd20 / dev-intel18; a GPU node also checks CUDA
#     cd ~/Capstone-MSU_Fruit/sam2_pipeline
#     bash hpcc/setup_env.sh
# Safe to run again: it only adds what is missing. Takes 10-20 minutes the first time.
set -eo pipefail          # (no -u: conda's activate scripts read unset variables)

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE_DIR/hpcc/config.sh"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"     # facebookresearch/sam2, 2024-12-15 (the latest)

echo "code:      $CODE_DIR"
echo "project:   $PROJECT_DIR"
echo "work dir:  $FRUIT_BASE"
echo "conda env: $ENV_DIR"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"

# 1. Conda environment with Python 3.12
if [ ! -x "$ENV_DIR/bin/python" ]; then
    conda create -y -p "$ENV_DIR" python=3.12 pip
fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip

# 2. PyTorch 2.14 built for CUDA 12.6: the newest build that still runs on the cluster's V100s (PyTorch's
#    CUDA 12.8+ builds dropped them) as well as its A100, L40S and H200 GPUs. Then the other packages.
#    To change the PyTorch build later: python -m pip uninstall -y torch torchvision, edit this line, re-run.
python -m pip install "torch==2.14.*" "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$CODE_DIR/requirements.txt"

# 3. Meta's SAM 2 code (inference + training), pinned to the commit the pipeline was tested with
mkdir -p "$FRUIT_BASE/weights"
if [ ! -f "$FRUIT_BASE/sam2_repo/sam2/build_sam.py" ]; then
    rm -rf "$FRUIT_BASE/sam2_repo"
    git init -q "$FRUIT_BASE/sam2_repo"
    git -C "$FRUIT_BASE/sam2_repo" remote add origin https://github.com/facebookresearch/sam2.git
    git -C "$FRUIT_BASE/sam2_repo" fetch -q --depth 1 origin "$SAM2_COMMIT"
    git -C "$FRUIT_BASE/sam2_repo" checkout -q FETCH_HEAD
fi

# 4. Weights, fetched now so jobs never need the internet: SAM 2.1 tiny (quick tests) and your chosen size,
#    the YOLOE detector with its text encoder, and the YOLO base model for train-detector.
cd "$FRUIT_BASE/weights"            # Ultralytics saves into the current folder; the job scripts run from here too
python "$CODE_DIR/fruit.py" --project "$PROJECT_DIR" download --sizes tiny "$SAM2_SIZE"

# 5. Jupyter kernel, so the notebook runs in OnDemand with this environment
python -m ipykernel install --user --name fruit_sam2 --display-name "Python (fruit_sam2)"

# 6. Check: a real GPU computation, not just "is a GPU visible"
python - <<'PY'
import torch, sys
print(f"python {sys.version.split()[0]} · torch {torch.__version__} (CUDA {torch.version.cuda})")
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    torch.cuda.synchronize(); float((x @ x).sum())         # fails with "no kernel image" on an unsupported GPU
    print("GPU works:", torch.cuda.get_device_name(0))
else:
    print("No GPU on this node (normal on a CPU dev node). Jobs get one from #SBATCH --gpus.")
PY
echo
echo "Setup done. Next (see README):"
echo "  1. put videos in $PROJECT_DIR/videos   (OnDemand Files ▸ Upload, rsync or Globus)"
echo "  2. source hpcc/activate.sh && python fruit.py clips    (cuts clips, splits train/val)"
echo "  3. label in OnDemand Jupyter: notebooks/MSU_FRUIT_SAM2_pipeline.ipynb, kernel 'Python (fruit_sam2)'"
echo "  4. sbatch hpcc/run.sb train --run grapes_v1 --steps 3000"
