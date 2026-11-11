# Use this pipeline in a terminal on the HPCC (a development node, or OnDemand's terminal):
#
#     cd ~/Capstone-MSU_Fruit/sam3_1_pipeline
#     source hpcc/activate.sh
#     python run.py video ../data/videos/row3.mp4
#
# Puts the environment that hpcc/setup_env.sh built first on the PATH, so `python` and `hf` are its own. The HPCC's
# default modules are cleared first: their Python 3.11 packages would otherwise be found before the environment's.
# Once per terminal. A development node stops a process after two hours of CPU time; longer runs go to the queue:
# sbatch hpcc/run.sb <script> <arguments>.
SAM31_ENV_DIR="${SAM31_ENV_DIR:-$HOME/envs/fruit_sam3_1}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts
if [ -x "$SAM31_ENV_DIR/bin/python" ]; then
    module purge > /dev/null 2>&1
    unset PYTHONPATH
    export PATH="$SAM31_ENV_DIR/bin:$PATH"
    # At most 8 CPU threads, unless set already: a development node stops a process after two hours of CPU time, and a
    # thread on each of its 192 cores uses that up in minutes.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(n=$(nproc); echo $((n < 8 ? n : 8)))}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    echo "fruit_sam3_1 environment active · $(python --version 2>&1) · $SAM31_ENV_DIR"
else
    echo "No environment at $SAM31_ENV_DIR. Build it once, on a development node: bash hpcc/setup_env.sh" >&2
fi
