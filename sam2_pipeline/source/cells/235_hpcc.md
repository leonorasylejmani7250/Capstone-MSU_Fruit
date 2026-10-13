## Appendix E · The same pipeline on the MSU HPCC
The `sam2_pipeline/` folder of the team repository holds this notebook, `fruit_sam2.py` (the tool cells above as a module), `fruit.py` (a command line around it) and `hpcc/` (setup and job scripts). The README walks through it; the short version:

1. **Once, on a development node** (`ssh dev-amd20-v100`): `git clone -b Saitej https://github.com/moehocbaidi/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit`, then `cd ~/Capstone-MSU_Fruit/sam2_pipeline && bash hpcc/setup_env.sh`. That builds the conda environment, fetches the SAM 2 code and the weights, and registers a Jupyter kernel.
2. **Videos** go to `~/msu_fruit_project/videos` (OnDemand *Files ▸ Upload* for files under about 1 GB; Globus or `rsync` to `rsync.hpcc.msu.edu` for big ones). Then `python fruit.py clips` cuts clips and splits them.
3. **Label** in OnDemand: *Interactive Apps ▸ Jupyter*, Jupyter location *Conda Environment using Miniforge3 module* with the environment path `~/envs/fruit_sam2`, one GPU under *Advanced Options*. Open this notebook, set `DEMO` off and `PROJECT_DIR = "~/msu_fruit_project"` in cell A, run the setup and tools cells, then Step 3.
4. **Train and score as batch jobs**, from the code folder:
   ```bash
   sbatch hpcc/run.sb train --run grapes_v1 --steps 3000
   sbatch hpcc/run.sb evaluate
   sbatch hpcc/run.sb video videos/row3.mp4 --seconds 60 --weights models/sam2.1_base_plus_grapes_v1.pt
   ```
   Results land in the project folder, so this notebook (on OnDemand, or on Colab after copying the folder to Drive) shows them too.
