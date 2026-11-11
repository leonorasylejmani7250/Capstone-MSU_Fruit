#!/bin/bash --login
# One-time setup of the SAM 3 environment on the MSU HPCC.
# Run it on a development node, because those have internet access (compute nodes may not):
#     ssh dev-amd20-v100
#     cd ~/Capstone-MSU_Fruit/sam3_pipeline
#     bash hpcc/setup_env.sh
#     ~/envs/fruit_sam3/bin/hf auth login
#     ~/envs/fruit_sam3/bin/hf download facebook/sam3 sam3.pt --local-dir model
set -eo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3_ENV_DIR="${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
if [ ! -x "$SAM3_ENV_DIR/bin/python" ]; then
    conda create -y -p "$SAM3_ENV_DIR" python=3.12 pip
fi
conda activate "$SAM3_ENV_DIR"
python -m pip install --upgrade pip
# CUDA 12.6 build: the newest PyTorch build that still runs on the cluster's V100s as well as its newer GPUs.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$CODE_DIR/requirements.txt"
python -c "import sam3, sam3.train, torch; print('sam3 ok · torch', torch.__version__, '· CUDA', torch.cuda.is_available())"
echo "Next, once, with your approved Hugging Face account:"
echo "  $SAM3_ENV_DIR/bin/hf auth login"
echo "  $SAM3_ENV_DIR/bin/hf download facebook/sam3 sam3.pt --local-dir $CODE_DIR/model"
echo "Then:  sbatch hpcc/run.sb run.py clips"
