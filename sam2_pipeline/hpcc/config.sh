# Settings shared by every HPCC script. Edit once; the job scripts read this file.
# (Paths here are on the HPCC, not your laptop.)
#
# Who reads it: setup_env.sh (builds what these lines name), activate.sh and through it run.sb, the YOLO
# pipelines' activate.sh and run.sb (they borrow this environment when they have none of their own), and
# hpcc/run_all.sh and hpcc/count_rows.sh in the repository folder. Change a path here and all of them follow.
# After changing FRUIT_BASE or ENV_DIR, run `bash hpcc/setup_env.sh` again so that it exists in the new place.
# The notebook does not read this file: in OnDemand you pick the environment in the Jupyter form (LABELLING.md,
# step 1), and the notebook keeps its SAM 2 code and weights in ~/.fruit_sam2 unless FRUIT_BASE is set there.

# The project folder: videos/, dataset/ (frames + labels), predictions/, models/, results/, runs/.
# Default: the data/ folder of this clone of the repository, the same folder every pipeline uses. Home is backed
# up by snapshots and has a 100 GB quota. Scratch ($SCRATCH) is bigger but deletes files untouched for 45 days,
# so do not keep the only copy of your labels there.
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data"

# SAM 2 code, pretrained weights and temporary files (re-creatable; setup_env.sh fills it).
export FRUIT_BASE="$HOME/.fruit_sam2"

# The conda environment setup_env.sh creates.
export ENV_DIR="$HOME/envs/fruit_sam2"

# Default SAM 2.1 size: tiny | small | base_plus | large
export SAM2_SIZE="base_plus"
export OBJECT_NAME="grape cluster"

# Quieter progress bars in job logs.
export TQDM_MININTERVAL=30
export MPLBACKEND=Agg
