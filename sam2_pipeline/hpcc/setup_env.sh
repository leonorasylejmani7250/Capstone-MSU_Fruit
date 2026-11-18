#!/bin/bash --login
# Builds everything the SAM 2.1 pipeline needs on the MSU HPCC, once per person: conda environment, SAM 2 code,
# pretrained weights, Jupyter kernel. hpcc/setup.sh in the repository folder runs it; it can also be run by
# itself, on a development node, because those have internet access (compute nodes may not):
#
#     ssh dev-amd24-h200
#     cd ~/Capstone-MSU_Fruit/sam2_pipeline
#     bash hpcc/setup_env.sh
#
# Any development node will do; one with GPUs, like dev-amd24-h200, lets step 6 test the GPU as well.
#
# WHAT IT MAKES, all of it in your HPCC home folder (~/envs/fruit_sam2 and ~/.fruit_sam2 are named in hpcc/config.sh)
#   ~/envs/fruit_sam2                                 the conda environment: Python 3.12, PyTorch, requirements.txt
#   ~/.fruit_sam2/sam2_repo                           Meta's SAM 2 code, one fixed version
#   ~/.fruit_sam2/checkpoints, ~/.fruit_sam2/weights  pretrained weights: SAM 2.1, the YOLOE detector, YOLO26
#   ~/.local/share/jupyter/kernels/fruit_sam2         the Jupyter kernel "Python (fruit_sam2)"
# It also makes the folders the pipeline writes to in the repository's data/ folder, where they are missing.
# The YOLO26 and YOLO11 pipelines run in this environment too (their hpcc/activate.sh and hpcc/run.sb use it).
#
# It edits no login file, submits no job and trains nothing; the modules and the environment it loads are
# gone again when it ends. Safe to run again: it only adds what is missing.
# Takes 10-20 minutes the first time.
#
# Each numbered step below says what it does, why, and what to edit to change it.
set -eo pipefail          # stop at the first command that fails (no -u: conda's activate scripts read unset variables)

# Where things go. CODE_DIR is this pipeline's folder, worked out from where this file is, so the script works
# from any folder. config.sh sets PROJECT_DIR (the repository's data/ folder), FRUIT_BASE (~/.fruit_sam2),
# ENV_DIR (~/envs/fruit_sam2) and SAM2_SIZE (base_plus). To put the environment or the weights somewhere else,
# edit config.sh and not this file: activate.sh, run.sb and the scripts in the repository's hpcc/ folder read it too.
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE_DIR/hpcc/config.sh"
# The version of Meta's SAM 2 code that step 3 fetches. source/cells/030_setup.py holds the same line for the
# notebook. To move to another version: change both, run `python source/build.py`, delete ~/.fruit_sam2/sam2_repo
# (step 3 only fetches when that folder is missing) and run this script again.
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"     # facebookresearch/sam2, 2024-12-15 (the latest)

echo "code:      $CODE_DIR"
echo "project:   $PROJECT_DIR"
echo "work dir:  $FRUIT_BASE"
echo "conda env: $ENV_DIR"

# The HPCC offers its software as "modules". The first line unloads the ones every login starts with: Miniforge3
# does not load next to them, and their Python 3.11 packages would get mixed into ours. The second makes conda
# available, the third makes `conda activate` work inside a script. All three hold only until this script ends.
module purge
module load Miniforge3
eval "$(conda shell.bash hook)"

# 1. The conda environment: a folder with its own Python 3.12 and pip, apart from the HPCC's Python and from the
#    other pipelines' environments. It is created only when it is not there yet (-p: at this path; -y: without
#    asking). `conda activate` then makes `python` and `pip` mean the environment's own for the rest of this script.
#    To change the Python version: edit python=3.12, delete the folder ~/envs/fruit_sam2, run this script again.
if [ ! -x "$ENV_DIR/bin/python" ]; then
    conda create -y -p "$ENV_DIR" python=3.12 pip
fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip

# 2. PyTorch 2.14 built for CUDA 12.6: the newest build that still runs on the cluster's V100s (PyTorch's
#    CUDA 12.8+ builds dropped them) as well as its A100, L40S and H200 GPUs. Then the other packages.
#    To change the PyTorch build later: after `source hpcc/activate.sh`, python -m pip uninstall -y torch torchvision;
#    then edit this line and re-run.
#    To add a package or fix its version: edit requirements.txt, which says what each package is for, and run
#    this script again. pip installs what is new; a package taken off the list stays until it is uninstalled.
python -m pip install "torch==2.14.*" "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$CODE_DIR/requirements.txt"

# 3. Meta's SAM 2 code (inference + training), pinned to one commit, in ~/.fruit_sam2/sam2_repo. The four git
#    lines download exactly that version without the project's history. Nothing is built or installed from it:
#    the pipeline reads the code straight from that folder. Skipped when the folder is already complete.
mkdir -p "$FRUIT_BASE/weights"
if [ ! -f "$FRUIT_BASE/sam2_repo/sam2/build_sam.py" ]; then
    rm -rf "$FRUIT_BASE/sam2_repo"
    git init -q "$FRUIT_BASE/sam2_repo"
    git -C "$FRUIT_BASE/sam2_repo" remote add origin https://github.com/facebookresearch/sam2.git
    git -C "$FRUIT_BASE/sam2_repo" fetch -q --depth 1 origin "$SAM2_COMMIT"
    git -C "$FRUIT_BASE/sam2_repo" checkout -q FETCH_HEAD
fi

# 4. Weights, fetched now so jobs never need the internet: SAM 2.1 tiny (quick tests) and your chosen size,
#    the YOLOE detector with its text encoder, and the YOLO base model for train-detector. The SAM 2.1 files go
#    to ~/.fruit_sam2/checkpoints, the others to ~/.fruit_sam2/weights; a file that is there is not fetched again.
#    To fetch other sizes or detectors: change SAM2_SIZE in config.sh, or the options of this line
#    (python fruit.py download --help lists them).
cd "$FRUIT_BASE/weights"            # Ultralytics saves into the current folder; the job scripts run from here too
python "$CODE_DIR/fruit.py" --project "$PROJECT_DIR" download --sizes tiny "$SAM2_SIZE"

# 5. Jupyter kernel, so the notebook runs in OnDemand with this environment. It is a small file in
#    ~/.local/share/jupyter/kernels/fruit_sam2 that tells Jupyter where this environment's Python is.
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
echo "Setup done. Next (from the repository folder; hpcc/README.md has each step):"
echo "  1. video into $PROJECT_DIR/videos              data/README.md"
echo "  2. label clips                                 LABELLING.md"
echo "  3. bash hpcc/run_all.sh grapes_v1              fine-tune every model and score them all"
echo "  4. bash hpcc/count_rows.sh                     count the clusters in every row video"
