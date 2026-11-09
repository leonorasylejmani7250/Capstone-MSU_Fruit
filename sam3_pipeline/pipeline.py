"""The names and folders of this pipeline. Every script in this folder imports them."""
import os
from pathlib import Path

NAME = 'sam3'                                       # this pipeline; names its folders under ../data/predictions and ../data/results
MODEL_NAME = 'SAM 3'                                # how the leaderboard names it
VERSION = 'sam3'                                    # which model Meta's `sam3` package builds: 'sam3.1' or 'sam3'
HF_REPO = 'facebook/sam3'                           # Meta's released weights on Hugging Face (gated)
RELEASE_FILE = 'sam3.pt'                            # the checkpoint file in that repository
ROOT = Path(__file__).resolve().parent              # this folder
# The project folder: the repository's data/ unless $PROJECT_DIR (sam2_pipeline/hpcc/config.sh) or --data says otherwise.
DATA = Path(os.environ.get('PROJECT_DIR') or ROOT.parent / 'data').expanduser()
MODEL_DIR = ROOT / 'model'                          # released and fine-tuned checkpoints (model/README.md)
RESULTS = DATA / 'results' / NAME                   # overlay videos and counts
