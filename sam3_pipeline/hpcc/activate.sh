# Use this pipeline in a terminal on the HPCC (a development node, or OnDemand's terminal):
#
#     cd ~/Capstone-MSU_Fruit/sam3_pipeline
#     source hpcc/activate.sh
#     python run.py video ../data/videos/row3.mp4
#
# Puts the environment that hpcc/setup_env.sh built first on the PATH, so `python` and `hf` are its own. The HPCC's
# default modules are cleared first: their Python 3.11 packages would otherwise be found before the environment's.
# Once per terminal. A development node stops a process after two hours; longer runs go to the queue:
# sbatch hpcc/run.sb <script> <arguments>.
SAM3_ENV_DIR="${SAM3_ENV_DIR:-$HOME/envs/fruit_sam3}"     # its own name: ENV_DIR belongs to the SAM 2.1 scripts
if [ -x "$SAM3_ENV_DIR/bin/python" ]; then
    module purge > /dev/null 2>&1
    unset PYTHONPATH
    export PATH="$SAM3_ENV_DIR/bin:$PATH"
    echo "fruit_sam3 environment active · $(python --version 2>&1) · $SAM3_ENV_DIR"
else
    echo "No environment at $SAM3_ENV_DIR. Build it once, on a development node: bash hpcc/setup_env.sh" >&2
fi
