#!/bin/bash
# Fine-tune every model on our labelled clips and score them all, as a chain of batch jobs on the MSU HPCC.
#
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/run_all.sh                       default run name "grapes_v1", 3000 SAM 2.1 steps, 100 YOLO epochs
#     bash hpcc/run_all.sh grapes_v2 5000 150    another run name, more steps and epochs
#
# What it submits (each with the pipeline's own hpcc/run.sb; `squeue --me` shows them, the logs are
# sam2_pipeline/fruit-<jobid>.out and sam3_pipeline/sam3-<jobid>.out):
#   1. SAM 2.1  fine-tune on our clips, from Meta's pretrained base_plus        fruit.py train --run <name> --steps <steps>
#   2. YOLO26   fine-tune on our clips, from yolo26_pipeline/model/best.pt      fruit.py train-detector --base ... --name yolo26_<name>
#   3. YOLO11   fine-tune on our clips, from yolo11_pipeline/model/best.pt      fruit.py train-detector --base ... --name yolo11_<name>
#   4. SAM 3.1  zero-shot masks for the val clips, no training                   run_sam3.py clips   (needs ~/envs/fruit_sam3)
#   5. leaderboard, after 1-4: every model in configs/lineup.yaml plus the four above, on the same val clips   fruit.py evaluate
#
# Before running it: the environments exist (sam2_pipeline/hpcc/setup_env.sh, sam3_pipeline/hpcc/setup_env.sh,
# `hf auth login`), and data/dataset holds labelled clips marked done (label in OnDemand Jupyter, or import labels
# with fruit.py import-coco / import-yolo + approve). Nothing here needs a GPU on the node you run it from.
set -eo pipefail
RUN="${1:-grapes_v1}"
STEPS="${2:-3000}"
EPOCHS="${3:-100}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM2="$REPO/sam2_pipeline"
SAM3="$REPO/sam3_pipeline"
YOLO26_MODEL="$REPO/yolo26_pipeline/model/best.pt"
YOLO11_MODEL="$REPO/yolo11_pipeline/model/best.pt"

source "$SAM2/hpcc/config.sh"
unset ENV_DIR                       # the SAM 2.1 environment; the other pipelines' jobs pick their own
echo "repository: $REPO"
echo "project folder (labelled clips, results): $PROJECT_DIR"
if [ ! -f "$PROJECT_DIR/dataset/clips.json" ]; then
    echo "No labelled clips in $PROJECT_DIR/dataset. Label clips first (sam2_pipeline/README.md, section 2)." >&2
    exit 1
fi

submit() {    # submit <folder> <sbatch options...> -- <job arguments...>; prints the job id
    local folder="$1"; shift
    local out
    out=$(cd "$folder" && sbatch --parsable "$@")
    echo "${out%%;*}"
}

cd "$SAM2"
J_SAM2=$(submit "$SAM2" hpcc/run.sb train --run "$RUN" --steps "$STEPS")
echo "job $J_SAM2: SAM 2.1  fine-tune on our clips from Meta's pretrained base_plus ($STEPS steps) → $PROJECT_DIR/models/sam2.1_base_plus_$RUN.pt"

J_Y26=$(submit "$SAM2" --time=02:00:00 hpcc/run.sb train-detector --base "$YOLO26_MODEL" --epochs "$EPOCHS" --name "yolo26_$RUN")
echo "job $J_Y26: YOLO26   fine-tune on our clips from yolo26_pipeline/model/best.pt ($EPOCHS epochs) → $PROJECT_DIR/runs/yolo26_$RUN/weights/best.pt"

J_Y11=$(submit "$SAM2" --time=02:00:00 hpcc/run.sb train-detector --base "$YOLO11_MODEL" --epochs "$EPOCHS" --name "yolo11_$RUN")
echo "job $J_Y11: YOLO11   fine-tune on our clips from yolo11_pipeline/model/best.pt ($EPOCHS epochs) → $PROJECT_DIR/runs/yolo11_$RUN/weights/best.pt"

DEPS="$J_SAM2:$J_Y26:$J_Y11"
if [ -x "${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}/bin/python" ]; then
    J_SAM3=$(submit "$SAM3" hpcc/run.sb clips)
    echo "job $J_SAM3: SAM 3.1  zero-shot masks for the val clips (no training) → $PROJECT_DIR/predictions/sam3_text"
    DEPS="$DEPS:$J_SAM3"
else
    echo "SAM 3.1  NOT submitted: no environment at ~/envs/fruit_sam3. Make it once with sam3_pipeline/hpcc/setup_env.sh (+ hf auth login), then run this script again."
fi

J_EVAL=$(submit "$SAM2" --dependency=afterany:$DEPS hpcc/run.sb evaluate)
echo "job $J_EVAL: leaderboard, once the others finish → $PROJECT_DIR/results/leaderboard_val.csv"
echo
echo "Watch: squeue --me      Logs: $SAM2/fruit-<jobid>.out      Cancel all: scancel $J_SAM2 $J_Y26 $J_Y11 ${J_SAM3:-} $J_EVAL"
