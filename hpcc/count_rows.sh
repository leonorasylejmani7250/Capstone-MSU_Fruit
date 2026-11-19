#!/bin/bash
# Count the clusters in every row video with every model, as batch jobs on the MSU HPCC.
#
#    cd ~/Capstone-MSU_Fruit
#    bash hpcc/count_rows.sh
#    bash hpcc/count_rows.sh 2026/Grape--DJIOsmo
#
# Without an argument it counts every video under data/videos; with one, a folder or a
# single video under data/videos.
#
# It submits one job per pipeline, each with the model that pipeline ships in its model/
# folder, then one job that writes the table. Totals land in
# data/results/cluster_counts.csv (one line per video and model) and
# cluster_counts_by_video.csv (one row per video, one column per model). A video that
# already has its total is skipped, so the script can be run again after new video
# arrives or when a job ran out of time.
#
# hpcc/README.md, step 5, describes the jobs.

# Stop at the first command that fails.
set -eo pipefail
WHAT="${1:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/sam2_pipeline/hpcc/config.sh"
SAM2_ENV="$ENV_DIR"

# sbatch hands this shell's variables to the jobs, and each pipeline's job picks its own
# environment.
unset ENV_DIR
VIDEOS="$PROJECT_DIR/videos${WHAT:+/$WHAT}"
if [ ! -e "$VIDEOS" ]; then
    echo "No such video or folder: $VIDEOS" >&2
    exit 1
fi
echo "videos: $VIDEOS"

#    submit <folder> <sbatch options...> <job script and arguments...>
# Submits a job from that folder and prints its job id.
submit() {
    local folder="$1"
    shift
    local out
    out=$(cd "$folder" && sbatch --parsable "$@")
    echo "${out%%;*}"
}

# Every job asks for 4 hours: jobs of that length usually start within minutes, longer
# ones wait.
J=$(submit "$REPO/sam2_pipeline" --time=04:00:00 hpcc/run.sb video "$VIDEOS")
echo "job $J: SAM 2.1  YOLO26 finds the clusters, SAM 2.1 follows them"
DEPS="$J"

for name in yolo26 yolo11; do
    var="$(echo "$name" | tr a-z A-Z)_ENV_DIR"
    env="$HOME/envs/fruit_$name"

    # Without an environment of its own, a YOLO pipeline runs in the SAM 2.1
    # environment.
    [ -x "$env/bin/python" ] || env="$SAM2_ENV"
    flag=--video-dir
    [ -f "$VIDEOS" ] && flag=--video
    J=$(submit "$REPO/${name}_pipeline" --time=04:00:00 --export="ALL,$var=$env" \
        hpcc/run.sb video.py "$flag" "$VIDEOS")
    echo "job $J: ${name^^}   tracker ids"
    DEPS="$DEPS:$J"
done

for name in sam3 sam3_1; do
    label="SAM 3  "
    [ "$name" = sam3_1 ] && label="SAM 3.1"
    env="$HOME/envs/fruit_$name"
    if [ ! -x "$env/bin/python" ]; then
        echo "$label  NOT submitted: no environment at $env" \
            "(${name}_pipeline/hpcc/setup_env.sh)"
        continue
    fi
    J=$(submit "$REPO/${name}_pipeline" --time=04:00:00 hpcc/run.sb run.py video \
        "$VIDEOS")
    echo "job $J: $label  text prompt \"grape cluster\""
    DEPS="$DEPS:$J"
done

J=$(submit "$REPO/sam2_pipeline" --dependency=afterany:$DEPS --time=00:15:00 \
    hpcc/run.sb counts)
echo "job $J: the table, once the others finish →" \
    "$PROJECT_DIR/results/cluster_counts_by_video.csv"
echo
echo "Watch: squeue --me      Totals so far: cd sam2_pipeline && source" \
    "hpcc/activate.sh && python fruit.py counts"
