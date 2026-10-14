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
#@markdown `PROJECT_DIR`: leave empty in Colab. On the MSU HPCC (OnDemand Jupyter) put the same folder
#@markdown as `PROJECT_DIR` in `hpcc/config.sh`, e.g. `~/msu_fruit_project`, so the notebook and the
#@markdown batch jobs share one project.
