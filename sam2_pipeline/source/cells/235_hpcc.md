## Appendix E · The same pipeline on the MSU HPCC
The `sam2_pipeline/` folder of the team repository holds this notebook, `fruit_sam2.py` (the tool cells above as a module), `fruit.py` (a command line around it) and `hpcc/` (setup and job scripts). The repository's `hpcc/README.md` walks through it; the short version:

1. **Once, on a development node** (`ssh dev-amd20-v100`): `git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit`, then `cd ~/Capstone-MSU_Fruit && bash hpcc/setup.sh`. That links the team's shared video and dataset folders, builds the conda environments, fetches the SAM 2 code and the weights, and registers a Jupyter kernel.
2. **Videos** go to the team's folder on the research space (`data/README.md`, "Video from SharePoint").
3. **Label** in OnDemand (`LABELLING.md`): *Interactive Apps ▸ Jupyter*, Jupyter location *Conda Environment using Miniforge3 module* with the environment path `~/envs/fruit_sam2`, one GPU under *Advanced Options*. Open this notebook, set `DEMO` off and `PROJECT_DIR = "~/Capstone-MSU_Fruit/data"` in cell A, then run everything above Step 3b and Step 3b itself.
4. **Train, score and count as batch jobs**, from the repository folder:
   ```bash
   bash hpcc/run_all.sh grapes_v1
   bash hpcc/count_rows.sh
   ```
   Results land in the project folder (`data/` in the repository), so this notebook (on OnDemand, or on Colab after copying the folder to Drive) shows them too.
