#!/bin/bash --login
# One-time setup of the SAM 3.1 environment on the MSU HPCC (untested; modelled on sam2_pipeline/hpcc/setup_env.sh).
# Run it on a development node, because those have internet access (compute nodes may not):
#     ssh dev-amd20-v100
#     cd ~/Capstone-MSU_Fruit/sam3_pipeline
#     bash hpcc/setup_env.sh
#     hf auth login
set -eo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/envs/fruit_sam3}"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
if [ ! -x "$ENV_DIR/bin/python" ]; then
    conda create -y -p "$ENV_DIR" python=3.12 pip
fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip
# CUDA 12.6 build: the newest PyTorch build that still runs on the cluster's V100s as well as its newer GPUs.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$CODE_DIR/requirements.txt"
python -c "import sam3, torch; print('sam3 ok · torch', torch.__version__, '· CUDA', torch.cuda.is_available())"
echo "Next: hf auth login   (once; needs your approved Hugging Face account), then: sbatch hpcc/run.sb clips"
