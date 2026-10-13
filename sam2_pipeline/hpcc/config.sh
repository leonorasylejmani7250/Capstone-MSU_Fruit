# Settings shared by every HPCC script. Edit once; the job scripts read this file.
# (Paths here are on the HPCC, not your laptop.)

# Your project: videos/, dataset/ (frames + labels), predictions/, models/, results/, runs/.
# Home is backed up by snapshots and has a 100 GB quota. Scratch ($SCRATCH) is bigger but deletes files
# untouched for 45 days, so do not keep the only copy of your labels there.
export PROJECT_DIR="$HOME/msu_fruit_project"

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
