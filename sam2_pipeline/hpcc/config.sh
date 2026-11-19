# Settings shared by every HPCC script of the repository. The paths are on the HPCC.
#
# setup_env.sh, activate.sh and run.sb of this folder read this file, and so do the YOLO
# pipelines' scripts, hpcc/run_all.sh and hpcc/count_rows.sh. After changing FRUIT_BASE
# or ENV_DIR, run `bash hpcc/setup_env.sh` again so that they exist in the new place.
# The notebook does not read this file.

# The project folder: videos/, dataset/, predictions/, models/, results/, runs/.
# Default: the data/ folder of this clone, which every pipeline uses. Do not move it to
# $SCRATCH, which deletes files left untouched for 45 days.
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data"

# Meta's SAM 2 code, the pretrained weights and temporary files. setup_env.sh fills it,
# and all of it can be fetched again.
export FRUIT_BASE="$HOME/.fruit_sam2"

# The conda environment that setup_env.sh creates.
export ENV_DIR="$HOME/envs/fruit_sam2"

# The default SAM 2.1 size: tiny, small, base_plus or large.
export SAM2_SIZE="base_plus"
export OBJECT_NAME="grape cluster"

# Quieter progress bars in job logs, and plots without a display.
export TQDM_MININTERVAL=30
export MPLBACKEND=Agg
