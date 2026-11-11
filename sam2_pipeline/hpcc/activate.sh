# Use the pipeline in an interactive HPCC shell (a dev node, or OnDemand's terminal):
#
#     cd ~/Capstone-MSU_Fruit/sam2_pipeline
#     source hpcc/activate.sh
#     python fruit.py status
#
# Loads conda, activates the environment setup_env.sh built, and reads hpcc/config.sh (PROJECT_DIR, by default
# the repository's data/ folder, etc.).
# Light commands (status, clips, counts, import-coco, import-yolo, approve, export-yolo) are fine on a dev node;
# send GPU work (train, evaluate, video, train-detector) to the queue with `sbatch hpcc/run.sb ...`.
# `check` takes a minute or two: `bash --login hpcc/run.sb check <video>` runs it on the dev node, without the queue.
_fruit_hpcc="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_fruit_hpcc/config.sh"
module purge > /dev/null 2>&1
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
# At most 8 CPU threads, unless set already: a development node stops a process after two hours of CPU time, and a
# thread on each of its 192 cores uses that up in minutes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(n=$(nproc); echo $((n < 8 ? n : 8)))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
echo "fruit_sam2 environment active · project folder: $PROJECT_DIR"
unset _fruit_hpcc
