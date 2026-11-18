#!/bin/bash --login
# Builds the SAM 3.1 environment on the MSU HPCC, once per person. hpcc/setup.sh in the repository folder runs it;
# it can also be run by itself, on a development node, because those have internet access (compute nodes may not):
#     ssh dev-amd24-h200
#     cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
#     bash hpcc/setup_env.sh
# and afterwards, once, to get Meta's weights (the script prints these two lines; hpcc/README.md, step 1):
#     ~/envs/fruit_sam3_1/bin/hf auth login
#     ~/envs/fruit_sam3_1/bin/hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir model
#
# WHAT IT MAKES: one folder, ~/envs/fruit_sam3_1, a conda environment with Python 3.12, PyTorch and the packages
# in requirements.txt (Meta's sam3 package, which holds SAM 3.1 as well, with its trainer). It edits no login
# file, submits no job, trains nothing, does not log in to Hugging Face and downloads no weights. The modules
# and the environment it loads are gone again when it ends. Safe to run again: the environment is kept and its
# packages are checked against requirements.txt.
#
# SAM 3 has an environment of its own, ~/envs/fruit_sam3, built the same way by sam3_pipeline/hpcc/setup_env.sh.
# Every pipeline keeps its own, so that the packages of one can be changed without touching another.
#
# TO CHANGE IT
#   a package            edit requirements.txt (it says what each line is for), run this script again
#   Meta's sam3 code     requirements.txt says how to fix its version or bring in the newest: this script alone
#                        keeps the copy that is installed
#   the Python version   edit python=3.12 below, delete the folder ~/envs/fruit_sam3_1, run this script again
#   the PyTorch build    after `source hpcc/activate.sh`: python -m pip uninstall -y torch torchvision; then edit
#                        the --index-url line below and run this script again
#   where it lives       the default path below is also in activate.sh and run.sb of this folder and in
#                        hpcc/run_all.sh and hpcc/count_rows.sh of the repository folder: change all of them
set -eo pipefail          # stop at the first command that fails (no -u: conda's activate scripts read unset variables)
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # this pipeline's folder, wherever this is run from
SAM31_ENV_DIR="${SAM31_ENV_DIR:-$HOME/envs/fruit_sam3_1}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts

# The HPCC offers its software as "modules". The first line unloads the ones every login starts with: Miniforge3
# does not load next to them, and their Python 3.11 packages would get mixed into ours. The second makes conda
# available, the third makes `conda activate` work inside a script. All three hold only until this script ends.
module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
# 1. The conda environment: a folder with its own Python and pip, created only when it is not there yet
#    (-p: at this path; -y: without asking). Python 3.12 is what Meta asks for with its sam3 package.
#    `conda activate` then makes `python` and `pip` mean the environment's own for the rest of this script.
if [ ! -x "$SAM31_ENV_DIR/bin/python" ]; then
    conda create -y -p "$SAM31_ENV_DIR" python=3.12 pip
fi
conda activate "$SAM31_ENV_DIR"
python -m pip install --upgrade pip
# 2. PyTorch, first, so that pip keeps this build for every package that needs PyTorch. cu126 is the
#    CUDA 12.6 build: the newest PyTorch build that still runs on the cluster's V100s as well as its newer GPUs.
#    No version is named, so pip takes the newest PyTorch that index has on the day the environment is built;
#    write "torch==2.14.*" as sam2_pipeline/hpcc/setup_env.sh does to fix it.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 3. The other packages: Meta's sam3 package with its trainer, from GitHub, and what our own scripts need.
python -m pip install -r "$CODE_DIR/requirements.txt"
# 4. Check: the three imports fail here, and not later in a job, when a package is missing. The line it prints
#    ends with whether this node has a GPU that PyTorch can use.
python -c "import sam3, sam3.train, torch; print('sam3 ok · torch', torch.__version__, '· CUDA', torch.cuda.is_available())"
# 5. Nothing is done here: these lines only print what comes next. Meta's weights need your own approved
#    Hugging Face account, so the login and the download are steps you run yourself.
echo "Next, once, with your approved Hugging Face account:"
echo "  $SAM31_ENV_DIR/bin/hf auth login"
echo "  $SAM31_ENV_DIR/bin/hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir $CODE_DIR/model"
echo "Then:  sbatch hpcc/run.sb run.py clips"
