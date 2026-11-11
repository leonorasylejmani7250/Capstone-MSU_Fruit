# Use this pipeline in a terminal on the HPCC (a development node, or OnDemand's terminal):
#
#     cd ~/Capstone-MSU_Fruit/yolo11_pipeline
#     source hpcc/activate.sh
#     python evaluate.py
#
# Puts an environment with Ultralytics first on the PATH, so `python` is its own: this folder's (hpcc/setup_env.sh
# builds it), else the SAM 2.1 environment, which hpcc/setup.sh in the repository folder builds and which runs the
# YOLO scripts too. The HPCC's default modules are cleared first: their Python 3.11 packages would otherwise be found
# before the environment's. Once per terminal. A development node stops a process after two hours of CPU time;
# longer runs go to the queue: sbatch hpcc/run.sb <script> <arguments>.
YOLO11_ENV_DIR="${YOLO11_ENV_DIR:-$HOME/envs/fruit_yolo11}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts
if [ ! -x "$YOLO11_ENV_DIR/bin/python" ]; then
    YOLO11_ENV_DIR="$(source "$(dirname "${BASH_SOURCE[0]}")/../../sam2_pipeline/hpcc/config.sh" && echo "$ENV_DIR")"
fi
if [ -x "$YOLO11_ENV_DIR/bin/python" ]; then
    module purge > /dev/null 2>&1
    unset PYTHONPATH
    export PATH="$YOLO11_ENV_DIR/bin:$PATH"
    # At most 8 CPU threads, unless set already: a development node stops a process after two hours of CPU time, and a
    # thread on each of its 192 cores uses that up in minutes.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(n=$(nproc); echo $((n < 8 ? n : 8)))}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    echo "$(basename "$YOLO11_ENV_DIR") environment active for YOLO11 · $(python --version 2>&1) · $YOLO11_ENV_DIR"
else
    echo "No environment at $YOLO11_ENV_DIR. Build it once, on a development node: bash hpcc/setup.sh in ~/Capstone-MSU_Fruit" >&2
fi
