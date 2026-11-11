#@title A · Settings  { display-mode: "form" }
#@markdown **Run this first.** Leave `DEMO` on for your first run: it uses public sample grape
#@markdown clips so you can watch every step work before you add your own videos.

PROJECT_NAME  = "msu_fruit_sam2"  #@param {type:"string"}
DEMO          = True              #@param {type:"boolean"}
SAVE_TO_DRIVE = True              #@param {type:"boolean"}
SAM2_SIZE     = "base_plus"       #@param ["tiny", "small", "base_plus", "large"]
OBJECT_NAME   = "grape cluster"   #@param {type:"string"}
PROJECT_DIR   = ""                #@param {type:"string"}

#@markdown ---
#@markdown `SAVE_TO_DRIVE` keeps videos, labels and trained weights in *My Drive / PROJECT_NAME* so a
#@markdown Colab disconnect does not erase your labelling. The demo never touches Drive.
#@markdown
#@markdown `PROJECT_DIR`: leave empty in Colab. In the team repository the project folder is its `data/`
#@markdown folder: on a laptop set `PROJECT_DIR = "../../data"` (from `sam2_pipeline/notebooks/`); on the
#@markdown MSU HPCC (OnDemand Jupyter) `"~/Capstone-MSU_Fruit/data"`, the same folder `hpcc/config.sh`
#@markdown gives the batch jobs, so the notebook and the jobs share one project. With `DEMO` on, the demo
#@markdown uses `<PROJECT_DIR>/msu_fruit_sam2_demo` instead (git ignores it), never the real project.
