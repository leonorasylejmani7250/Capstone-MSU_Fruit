#!/bin/bash --login
# Build the SAM 3 environment on the MSU HPCC, once per person. hpcc/setup.sh in the
# repository folder runs this script. It can also be run by itself, on a development
# node (those have internet access):
#
#    ssh dev-amd24-h200
#    cd ~/Capstone-MSU_Fruit/sam3_pipeline
#    bash hpcc/setup_env.sh
#
# It makes one folder, ~/envs/fruit_sam3: a conda environment with Python 3.12, PyTorch
# and the packages of requirements.txt (Meta's sam3 package with its trainer). It is
# safe to run again. Meta's weights are not downloaded here: the script ends by printing
# the two `hf` lines to run with an approved Hugging Face account.
#
# How to change the environment: hpcc/README.md in the repository folder, "The
# environments".

# Stop at the first command that fails. (No -u: conda's activate scripts read unset
# variables.)
set -eo pipefail

# This pipeline's folder, wherever this script is run from.
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The environment has a variable of its own: ENV_DIR belongs to the SAM 2.1 scripts.
SAM3_ENV_DIR="${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}"

# Make conda available: unload the modules every login starts with, load Miniforge3, and
# let `conda activate` work inside this script.
module purge
module load Miniforge3
eval "$(conda shell.bash hook)"

# 1. The conda environment, created only when it is not there yet. Python 3.12 is what
# Meta asks for with SAM 3.
if [ ! -x "$SAM3_ENV_DIR/bin/python" ]; then
    conda create -y -p "$SAM3_ENV_DIR" python=3.12 pip
fi
conda activate "$SAM3_ENV_DIR"
python -m pip install --upgrade pip

# 2. PyTorch first, so that pip keeps this build for every package that needs it. cu126
# is the CUDA 12.6 build, the newest that still runs on the cluster's V100s.
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# 3. The other packages: Meta's sam3 package with its trainer, and what our own scripts
# need.
python -m pip install -r "$CODE_DIR/requirements.txt"

# 4. Check: a missing package fails here and not later in a job. The line it prints ends
# with whether this node has a GPU that PyTorch can use.
python -c "
import sam3, sam3.train, torch
print('sam3 ok · torch', torch.__version__, '· CUDA', torch.cuda.is_available())
"

# 5. What comes next. Meta's weights need your own approved Hugging Face account, so the
# login and the download are steps you run yourself.
echo "Next, once, with your approved Hugging Face account:"
echo "  $SAM3_ENV_DIR/bin/hf auth login"
echo "  $SAM3_ENV_DIR/bin/hf download facebook/sam3 sam3.pt --local-dir $CODE_DIR/model"
echo "Then:  sbatch hpcc/run.sb run.py clips"
