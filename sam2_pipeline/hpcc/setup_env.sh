#!/bin/bash --login
# Build everything the SAM 2.1 pipeline needs on the MSU HPCC, once per person: the
# conda environment, Meta's SAM 2 code, the pretrained weights and the Jupyter kernel.
# hpcc/setup.sh in the repository folder runs this script. It can also be run by itself,
# on a development node (those have internet access):
#
#    ssh dev-amd24-h200
#    cd ~/Capstone-MSU_Fruit/sam2_pipeline
#    bash hpcc/setup_env.sh
#
# It takes 10 to 20 minutes the first time and is safe to run again: it only adds what
# is missing. The YOLO26 and YOLO11 pipelines run in this environment too.
#
# What it leaves in the account, what it does not do and how to change the environment:
# hpcc/README.md in the repository folder, "What setup.sh does" and "The environments".
# The paths it uses are set in hpcc/config.sh.

# Stop at the first command that fails. (No -u: conda's activate scripts read unset
# variables.)
set -eo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE_DIR/hpcc/config.sh"

# The version of Meta's SAM 2 code that step 3 fetches (facebookresearch/sam2,
# 2024-12-15). source/cells/030_setup.py holds the same line for the notebook.
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"

echo "code:      $CODE_DIR"
echo "project:   $PROJECT_DIR"
echo "work dir:  $FRUIT_BASE"
echo "conda env: $ENV_DIR"

# Make conda available: unload the modules every login starts with, load Miniforge3, and
# let `conda activate` work inside this script.
module purge
module load Miniforge3
eval "$(conda shell.bash hook)"

# 1. The conda environment, created only when it is not there yet.
if [ ! -x "$ENV_DIR/bin/python" ]; then
    conda create -y -p "$ENV_DIR" python=3.12 pip
fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip

# 2. PyTorch 2.14 built for CUDA 12.6, the newest build that still runs on the cluster's
# V100s, then the packages of requirements.txt.
python -m pip install "torch==2.14.*" "torchvision==0.29.*" \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$CODE_DIR/requirements.txt"

# 3. Meta's SAM 2 code at the commit named above, without its history. The pipeline
# reads the code straight from that folder. Skipped when the folder is complete.
mkdir -p "$FRUIT_BASE/weights"
if [ ! -f "$FRUIT_BASE/sam2_repo/sam2/build_sam.py" ]; then
    rm -rf "$FRUIT_BASE/sam2_repo"
    git init -q "$FRUIT_BASE/sam2_repo"
    git -C "$FRUIT_BASE/sam2_repo" remote add origin \
        https://github.com/facebookresearch/sam2.git
    git -C "$FRUIT_BASE/sam2_repo" fetch -q --depth 1 origin "$SAM2_COMMIT"
    git -C "$FRUIT_BASE/sam2_repo" checkout -q FETCH_HEAD
fi

# 4. The weights, fetched now so that jobs never need the internet: SAM 2.1 tiny and the
# chosen size, the YOLOE detector and the YOLO base model. Ultralytics saves into the
# current folder, which is also where the job scripts run.
cd "$FRUIT_BASE/weights"
python "$CODE_DIR/fruit.py" --project "$PROJECT_DIR" download --sizes tiny "$SAM2_SIZE"

# 5. The Jupyter kernel, so that the notebook runs in OnDemand with this environment.
python -m ipykernel install --user --name fruit_sam2 \
    --display-name "Python (fruit_sam2)"

# 6. Check with a real GPU computation, not only whether a GPU is visible.
python - <<'PY'
import sys

import torch

print(
    f"python {sys.version.split()[0]} · torch {torch.__version__} "
    f"(CUDA {torch.version.cuda})"
)
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    torch.cuda.synchronize()
    # This fails with "no kernel image" on an unsupported GPU.
    float((x @ x).sum())
    print("GPU works:", torch.cuda.get_device_name(0))
else:
    print(
        "No GPU on this node (normal on a CPU dev node). "
        "Jobs get one from #SBATCH --gpus."
    )
PY
echo
echo "Setup done. Next (from the repository folder; hpcc/README.md has each step):"
echo "  1. video into $PROJECT_DIR/videos              data/README.md"
echo "  2. label clips                                 LABELLING.md"
echo "  3. bash hpcc/run_all.sh grapes_v1              fine-tune every model and" \
    "score them all"
echo "  4. bash hpcc/count_rows.sh                     count the clusters in every" \
    "row video"
