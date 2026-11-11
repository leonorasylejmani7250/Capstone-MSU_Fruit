#!/bin/bash
# Fine-tune every model on our labelled clips and score them all, as a chain of batch jobs on the MSU HPCC.
#
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/run_all.sh                       default run name "grapes_v1", 3000 SAM 2.1 steps, 100 YOLO epochs, 20 SAM 3 epochs
#     bash hpcc/run_all.sh grapes_v2 5000 150 30 another run name, more steps and epochs
#
# What it submits (each through the pipeline's own hpcc/run.sb; `squeue --me` shows them; the logs are
# sam2_pipeline/fruit-<jobid>.out, sam3_pipeline/sam3-<jobid>.out and sam3_1_pipeline/sam3_1-<jobid>.out):
#   SAM 2.1   fine-tune on our clips, from Meta's pretrained base_plus         fruit.py train --run <name> --steps <steps>
#   YOLO26    fine-tune on our clips, from yolo26_pipeline/model/best.pt       fruit.py train-detector --base ... --name yolo26_<name>
#   YOLO11    fine-tune on our clips, from yolo11_pipeline/model/best.pt       fruit.py train-detector --base ... --name yolo11_<name>
#   SAM 3     zero-shot masks for the val clips; fine-tune on our clips; masks with the fine-tuned model
#             run.py clips --checkpoint release · finetune.py --run <name> · run.py clips        (needs ~/envs/fruit_sam3)
#   SAM 3.1   the same three jobs                                                                (needs ~/envs/fruit_sam3_1)
#   leaderboard, after all of them: every model in configs/lineup.yaml plus everything above, on the same val clips
#             fruit.py evaluate
#
# Before running it: the environments exist (hpcc/setup.sh; SAM 3 and SAM 3.1 also need `hf auth login` and the downloaded
# release in their model/ folder), and data/dataset holds labelled clips marked done (LABELLING.md, or labels imported with
# fruit.py import-coco / import-yolo + approve). Nothing here needs a GPU on the node you run it from.
set -eo pipefail
RUN="${1:-grapes_v1}"
STEPS="${2:-3000}"
EPOCHS="${3:-100}"
SAM3_EPOCHS="${4:-20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM2="$REPO/sam2_pipeline"
YOLO26_MODEL="$REPO/yolo26_pipeline/model/best.pt"
YOLO11_MODEL="$REPO/yolo11_pipeline/model/best.pt"

source "$SAM2/hpcc/config.sh"
unset ENV_DIR                       # the SAM 2.1 environment; the other pipelines' jobs pick their own
echo "repository: $REPO"
echo "project folder (labelled clips, results): $PROJECT_DIR"
if [ ! -f "$PROJECT_DIR/dataset/clips.json" ]; then
    echo "No labelled clips in $PROJECT_DIR/dataset. Label clips first (LABELLING.md)." >&2
    exit 1
fi

submit() {    # submit <folder> <sbatch options...> <job script and arguments...>; prints the job id
    local folder="$1"; shift
    local out
    out=$(cd "$folder" && sbatch --parsable "$@")
    echo "${out%%;*}"
}

J_SAM2=$(submit "$SAM2" hpcc/run.sb train --run "$RUN" --steps "$STEPS")
echo "job $J_SAM2: SAM 2.1  fine-tune on our clips from Meta's pretrained base_plus ($STEPS steps) → $PROJECT_DIR/models/sam2.1_base_plus_$RUN.pt"

J_Y26=$(submit "$SAM2" --time=02:00:00 hpcc/run.sb train-detector --base "$YOLO26_MODEL" --epochs "$EPOCHS" --name "yolo26_$RUN")
echo "job $J_Y26: YOLO26   fine-tune on our clips from yolo26_pipeline/model/best.pt ($EPOCHS epochs) → $PROJECT_DIR/runs/yolo26_$RUN/weights/best.pt"

J_Y11=$(submit "$SAM2" --time=02:00:00 hpcc/run.sb train-detector --base "$YOLO11_MODEL" --epochs "$EPOCHS" --name "yolo11_$RUN")
echo "job $J_Y11: YOLO11   fine-tune on our clips from yolo11_pipeline/model/best.pt ($EPOCHS epochs) → $PROJECT_DIR/runs/yolo11_$RUN/weights/best.pt"

DEPS="$J_SAM2:$J_Y26:$J_Y11"
ALL="$J_SAM2 $J_Y26 $J_Y11"

sam3_jobs() {   # sam3_jobs <label> <folder> <environment> : zero-shot masks, fine-tune, masks with the fine-tuned model
    local label="$1" folder="$2" env="$3" name
    name="$(basename "$folder" _pipeline)"
    if [ ! -x "$env/bin/python" ]; then
        echo "$label  NOT submitted: no environment at $env. Make it once with $(basename "$folder")/hpcc/setup_env.sh (+ hf auth login, hf download), then run this script again."
        return
    fi
    local j_zero j_ft j_clips
    j_zero=$(submit "$folder" hpcc/run.sb run.py clips --checkpoint release)
    echo "job $j_zero: $label  zero-shot masks for the val clips (Meta's release, no training) → $PROJECT_DIR/predictions/${name}_text"
    j_ft=$(submit "$folder" --time=04:00:00 hpcc/run.sb finetune.py --run "$RUN" --epochs "$SAM3_EPOCHS")
    echo "job $j_ft: $label  fine-tune on our clips from Meta's release ($SAM3_EPOCHS epochs) → $folder/model/${name}_$RUN.pt"
    j_clips=$(submit "$folder" --dependency=afterok:$j_ft --kill-on-invalid-dep=yes hpcc/run.sb run.py clips)
    echo "job $j_clips: $label  masks for the val clips with the fine-tuned model, after job $j_ft → $PROJECT_DIR/predictions/${name}_$RUN"
    DEPS="$DEPS:$j_zero:$j_clips"
    ALL="$ALL $j_zero $j_ft $j_clips"
}
sam3_jobs "SAM 3  " "$REPO/sam3_pipeline" "${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}"
sam3_jobs "SAM 3.1" "$REPO/sam3_1_pipeline" "${SAM31_ENV_DIR:-$HOME/envs/fruit_sam3_1}"

J_EVAL=$(submit "$SAM2" --dependency=afterany:$DEPS hpcc/run.sb evaluate)
echo "job $J_EVAL: leaderboard, once the others finish → $PROJECT_DIR/results/leaderboard_val.csv"
echo
echo "Watch: squeue --me      Logs: $SAM2/fruit-<jobid>.out, $REPO/sam3_pipeline/sam3-<jobid>.out, $REPO/sam3_1_pipeline/sam3_1-<jobid>.out"
echo "Cancel all: scancel $ALL $J_EVAL"
