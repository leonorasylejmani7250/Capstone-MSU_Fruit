# Use the pipeline in an interactive HPCC shell (a dev node, or OnDemand's terminal):
#
#     cd ~/Capstone-MSU_Fruit/sam2_pipeline
#     source hpcc/activate.sh
#     python fruit.py status
#
# Loads conda, activates the environment setup_env.sh built, and reads hpcc/config.sh (PROJECT_DIR etc.).
# Light commands (status, clips, import-coco, import-yolo, approve, export-yolo) are fine on a dev node;
# send GPU work (train, evaluate, video, train-detector) to the queue with `sbatch hpcc/run.sb ...`.
_fruit_hpcc="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_fruit_hpcc/config.sh"
module purge > /dev/null 2>&1
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
echo "fruit_sam2 environment active · project folder: $PROJECT_DIR"
unset _fruit_hpcc
