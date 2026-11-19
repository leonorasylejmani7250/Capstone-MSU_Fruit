# Use the SAM 3 pipeline in a terminal on the HPCC (a development node, or OnDemand's
# terminal):
#
#    cd ~/Capstone-MSU_Fruit/sam3_pipeline
#    source hpcc/activate.sh
#    python run.py clips
#
# It puts the environment that hpcc/setup_env.sh built first on the PATH, so that
# `python` and `hf` are its own. That lasts until the terminal is closed, and no file is
# changed. Long runs go to the queue: `sbatch hpcc/run.sb <script> <arguments>`.

SAM3_ENV_DIR="${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}"
if [ -x "$SAM3_ENV_DIR/bin/python" ]; then
    # Unload the modules every login starts with: their Python packages would otherwise
    # be found before the environment's.
    module purge > /dev/null 2>&1
    unset PYTHONPATH
    export PATH="$SAM3_ENV_DIR/bin:$PATH"

    # At most 8 CPU threads, unless set already: a development node stops a process
    # after two hours of CPU time, and a thread on each of its cores uses that up in
    # minutes.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(n=$(nproc); echo $((n < 8 ? n : 8)))}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    echo "fruit_sam3 environment active · $(python --version 2>&1) · $SAM3_ENV_DIR"
else
    echo "No environment at $SAM3_ENV_DIR. Build it once, on a development node: bash" \
        "hpcc/setup_env.sh" >&2
fi
