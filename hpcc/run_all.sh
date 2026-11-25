#!/bin/bash
# Fine-tune every model on our labelled clips and score them all, as a chain of batch
# jobs on the MSU HPCC.
#
#    cd ~/Capstone-MSU_Fruit
#    bash hpcc/run_all.sh
#    bash hpcc/run_all.sh grapes_v2 5000 150 30
#
# The four arguments are the run name (default grapes_v1), the SAM 2.1 training steps
# (3000), the YOLO epochs (100) and the SAM 3 epochs (20).
#
# It submits one job per step through each pipeline's hpcc/run.sb: the fine-tuning of
# SAM 2.1 (two jobs: the second goes on where the first stopped if its 4 hours were not
# enough), of YOLO26 and of YOLO11; for SAM 3 and for SAM 3.1 the zero-shot masks, the
# fine-tuning and the masks of the fine-tuned model; then the leaderboard of every
# model. Before running it, the environments must exist (hpcc/setup.sh) and data/dataset
# must hold labelled clips marked done (LABELLING.md). Nothing here needs a GPU on the
# node it is run from. A run name whose jobs are still queued or running is not
# submitted again.
#
# hpcc/README.md, step 4, describes the jobs and their logs.

# Stop at the first command that fails.
set -eo pipefail
RUN="${1:-grapes_v1}"
STEPS="${2:-3000}"
EPOCHS="${3:-100}"
SAM3_EPOCHS="${4:-20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM2="$REPO/sam2_pipeline"
YOLO26_MODEL="$REPO/yolo26_pipeline/model/best.pt"
YOLO11_MODEL="$REPO/yolo11_pipeline/model/best.pt"

# config.sh sets PROJECT_DIR and ENV_DIR, the SAM 2.1 environment. ENV_DIR is unset
# again because sbatch hands this shell's variables to the jobs, and each pipeline's job
# picks its own environment.
source "$SAM2/hpcc/config.sh"
unset ENV_DIR
echo "repository: $REPO"
echo "project folder (labelled clips, results): $PROJECT_DIR"
if [ ! -f "$PROJECT_DIR/dataset/clips.json" ]; then
    echo "No labelled clips in $PROJECT_DIR/dataset. Label clips first" \
        "(LABELLING.md)." >&2
    exit 1
fi

# The ids of the jobs submitted for a run name are kept in runs/run_all_<name>.txt.
# While any of them is queued or running, submitting the name again would train a
# second time into the same folders, so the script stops.
JOBS="$PROJECT_DIR/runs/run_all_$RUN.txt"
if [ -f "$JOBS" ]; then
    QUEUED=$(squeue --me --noheader --format=%i)
    LIVE=""
    while read -r j; do
        if [ -n "$j" ] && grep -qx "$j" <<<"$QUEUED"; then
            LIVE="$LIVE $j"
        fi
    done <"$JOBS"
    if [ -n "$LIVE" ]; then
        echo "$RUN is already submitted and job(s)$LIVE are still queued or running." \
            "Wait for them, or cancel them first: scancel$LIVE" >&2
        exit 1
    fi
fi
mkdir -p "$PROJECT_DIR/runs"
: >"$JOBS"

#    submit <folder> <sbatch options...> <job script and arguments...>
# Submits a job from that folder, prints its job id and adds it to $JOBS.
submit() {
    local folder="$1"
    shift
    local out
    out=$(cd "$folder" && sbatch --parsable "$@")
    echo "${out%%;*}" | tee -a "$JOBS"
}

J_SAM2=$(submit "$SAM2" hpcc/run.sb train --run "$RUN" --steps "$STEPS")
echo "job $J_SAM2: SAM 2.1  fine-tune on our clips from Meta's pretrained base_plus" \
    "($STEPS steps) → $PROJECT_DIR/models/sam2.1_base_plus_$RUN.pt"

# The same line once more: a job gets 4 hours, and $STEPS steps on many clips can take
# longer. The second job goes on from the first one's last checkpoint; when the model is
# already there it ends at once.
J_SAM2_MORE=$(submit "$SAM2" "--dependency=afterany:$J_SAM2" hpcc/run.sb train \
    --run "$RUN" --steps "$STEPS")
echo "job $J_SAM2_MORE: SAM 2.1  the same training again, after job $J_SAM2: it goes on" \
    "where that job stopped if 4 hours were not enough"

J_Y26=$(submit "$SAM2" --time=02:00:00 hpcc/run.sb train-detector \
    --base "$YOLO26_MODEL" --epochs "$EPOCHS" --name "yolo26_$RUN")
echo "job $J_Y26: YOLO26   fine-tune on our clips from yolo26_pipeline/model/best.pt" \
    "($EPOCHS epochs) → $PROJECT_DIR/runs/yolo26_$RUN/weights/best.pt"

J_Y11=$(submit "$SAM2" --time=02:00:00 hpcc/run.sb train-detector \
    --base "$YOLO11_MODEL" --epochs "$EPOCHS" --name "yolo11_$RUN")
echo "job $J_Y11: YOLO11   fine-tune on our clips from yolo11_pipeline/model/best.pt" \
    "($EPOCHS epochs) → $PROJECT_DIR/runs/yolo11_$RUN/weights/best.pt"

# DEPS: the jobs the leaderboard waits for. ALL: every job, for the scancel line printed
# at the end.
DEPS="$J_SAM2_MORE:$J_Y26:$J_Y11"
ALL="$J_SAM2 $J_SAM2_MORE $J_Y26 $J_Y11"

#    sam3_jobs <label> <folder> <environment>
# The three jobs of one SAM 3 pipeline: zero-shot masks, fine-tuning, masks with the
# fine-tuned model.
sam3_jobs() {
    local label="$1" folder="$2" env="$3" name
    name="$(basename "$folder" _pipeline)"
    if [ ! -x "$env/bin/python" ]; then
        echo "$label  NOT submitted: no environment at $env. Make it once with" \
            "$(basename "$folder")/hpcc/setup_env.sh (+ hf auth login, hf download)," \
            "then run this script again."
        return
    fi
    local j_zero j_ft j_clips
    j_zero=$(submit "$folder" hpcc/run.sb run.py clips --checkpoint release)
    echo "job $j_zero: $label  zero-shot masks for the val clips (Meta's release, no" \
        "training) → $PROJECT_DIR/predictions/${name}_text"
    j_ft=$(submit "$folder" --time=04:00:00 hpcc/run.sb finetune.py --run "$RUN" \
        --epochs "$SAM3_EPOCHS")
    echo "job $j_ft: $label  fine-tune on our clips from Meta's release ($SAM3_EPOCHS" \
        "epochs) → $folder/model/${name}_$RUN.pt"
    j_clips=$(submit "$folder" --dependency=afterok:$j_ft --kill-on-invalid-dep=yes \
        hpcc/run.sb run.py clips)
    echo "job $j_clips: $label  masks for the val clips with the fine-tuned model," \
        "after job $j_ft → $PROJECT_DIR/predictions/${name}_$RUN"
    DEPS="$DEPS:$j_zero:$j_clips"
    ALL="$ALL $j_zero $j_ft $j_clips"
}
sam3_jobs "SAM 3  " "$REPO/sam3_pipeline" "${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}"
sam3_jobs "SAM 3.1" "$REPO/sam3_1_pipeline" "${SAM31_ENV_DIR:-$HOME/envs/fruit_sam3_1}"

J_EVAL=$(submit "$SAM2" --dependency=afterany:$DEPS hpcc/run.sb evaluate)
echo "job $J_EVAL: leaderboard, once the others finish →" \
    "$PROJECT_DIR/results/leaderboard_val.csv"
echo
echo "Watch: squeue --me      Logs: $SAM2/fruit-<jobid>.out," \
    "$REPO/sam3_pipeline/sam3-<jobid>.out, $REPO/sam3_1_pipeline/sam3_1-<jobid>.out"
echo "Cancel all: scancel $ALL $J_EVAL"
