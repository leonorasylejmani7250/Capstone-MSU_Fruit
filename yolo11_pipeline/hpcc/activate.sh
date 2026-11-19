# Use the YOLO11 pipeline in a terminal on the HPCC (a development node, or OnDemand's
# terminal):
#
#    cd ~/Capstone-MSU_Fruit/yolo11_pipeline
#    source hpcc/activate.sh
#    python evaluate.py
#
# It puts an environment with Ultralytics first on the PATH, so that `python` is its
# own: this folder's if hpcc/setup_env.sh has built it, else the SAM 2.1 environment,
# which runs the YOLO scripts too. That lasts until the terminal is closed, and no file
# is changed. Long runs go to the queue: `sbatch hpcc/run.sb <script> <arguments>`.

# This folder's own environment if it has been built, else the one whose place is
# ENV_DIR in sam2_pipeline/hpcc/config.sh.
YOLO11_ENV_DIR="${YOLO11_ENV_DIR:-$HOME/envs/fruit_yolo11}"
if [ ! -x "$YOLO11_ENV_DIR/bin/python" ]; then
    YOLO11_ENV_DIR="$(
        source "$(dirname "${BASH_SOURCE[0]}")/../../sam2_pipeline/hpcc/config.sh" \
            && echo "$ENV_DIR"
    )"
fi
if [ -x "$YOLO11_ENV_DIR/bin/python" ]; then
    # Unload the modules every login starts with: their Python packages would otherwise
    # be found before the environment's.
    module purge > /dev/null 2>&1
    unset PYTHONPATH
    export PATH="$YOLO11_ENV_DIR/bin:$PATH"

    # At most 8 CPU threads, unless set already: a development node stops a process
    # after two hours of CPU time, and a thread on each of its cores uses that up in
    # minutes.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(n=$(nproc); echo $((n < 8 ? n : 8)))}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    echo "$(basename "$YOLO11_ENV_DIR") environment active for YOLO11 ·" \
        "$(python --version 2>&1) · $YOLO11_ENV_DIR"
else
    echo "No environment at $YOLO11_ENV_DIR. Build it once, on a development node:" \
        "bash hpcc/setup.sh in ~/Capstone-MSU_Fruit" >&2
fi
