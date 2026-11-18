#!/bin/bash
# hpcc/setup.sh: gets one person's account on the MSU HPCC ready for the pipelines. Run it once, on a
# development node (those have internet access):
#
#     ssh dev-amd24-h200
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/setup.sh
#
# WHERE IT RUNS
#   On the HPCC, in the account of the person who runs it. Nothing is installed or changed on your own
#   computer: from there you only open the ssh connection.
#
# WHAT IT DOES
#   It runs four scripts, one after the other, and stops at the first one that fails. Each of them explains
#   its own steps in its comments.
#
#   1. hpcc/share_data.sh
#      Turns data/videos and data/dataset in this clone into links to the team's shared folders,
#      /mnt/research/MSU-Fruit/videos and /mnt/research/MSU-Fruit/dataset, so that everyone works on one
#      copy of the video and one set of clips and labels. You have to be a member of the MSU-Fruit research
#      space: if you are not, the script stops here and nothing has been changed.
#   2. sam2_pipeline/hpcc/setup_env.sh
#      Builds the conda environment ~/envs/fruit_sam2: Python 3.12, PyTorch and the packages listed in
#      sam2_pipeline/requirements.txt. The two YOLO pipelines run in this environment as well. It also
#      fetches Meta's SAM 2 code and the pretrained weights into ~/.fruit_sam2, and registers the Jupyter
#      kernel "Python (fruit_sam2)", so that a notebook can run in this environment.
#   3. sam3_pipeline/hpcc/setup_env.sh
#      Builds the conda environment ~/envs/fruit_sam3: Python 3.12, PyTorch and Meta's sam3 package.
#   4. sam3_1_pipeline/hpcc/setup_env.sh
#      Builds ~/envs/fruit_sam3_1 the same way, for the SAM 3.1 pipeline.
#
# WHAT IT LEAVES IN YOUR ACCOUNT (the home folder has a 100 GB quota)
#   ~/envs/fruit_sam2, ~/envs/fruit_sam3, ~/envs/fruit_sam3_1   the three environments, about 22 GB together
#   ~/.fruit_sam2/                                              Meta's SAM 2 code and the weights, about 1 GB
#   ~/.local/share/jupyter/kernels/fruit_sam2/                  the Jupyter kernel, a few small files
#   ~/.cache/pip                                                pip's copy of its downloads, about 3 GB, safe to delete
#   in this clone, under data/                                  the two links, the empty folders predictions/ and models/
#   The programs it runs (conda, pip, Ultralytics, matplotlib) also keep a few small settings and cache files
#   under ~/.conda, ~/.config and ~/.cache.
#
# WHAT IT DOES NOT DO
#   It needs no administrator rights and asks no questions. It does not edit ~/.bashrc or any other login
#   file: the modules and the environment a script loads are gone again when that script ends. It submits
#   no batch jobs and trains nothing. It does not log in to Hugging Face and does not download Meta's SAM 3
#   weights: those are the two `hf` blocks that follow it in hpcc/README.md.
#   In a fresh clone it moves and deletes nothing. Only when data/videos or data/dataset already hold files
#   does step 1 move them to the shared folder (hpcc/share_data.sh says exactly how).
#
# RUNNING IT AGAIN is safe: an environment that exists is kept and its packages are checked against the
# list, a file that is there is not downloaded again, a link that is right is left alone.
#
# TO REMOVE WHAT IT MADE, delete the folders listed above. The shared video and labels stay: they are in the
# research space, not in your account.
#
# TO CHANGE IT: the PIPELINES line below names the pipelines whose environment is built. Each pipeline's
# hpcc/setup_env.sh says how to change that environment; hpcc/README.md ("The environments") has the overview.
set -eo pipefail          # stop at the first command that fails
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # the repository folder, wherever this is run from

# One environment for each folder named here; every one of them has an hpcc/setup_env.sh. The two YOLO
# pipelines are not in the list because they run in the SAM 2.1 environment.
PIPELINES=(sam2_pipeline sam3_pipeline sam3_1_pipeline)
STEPS=$(( ${#PIPELINES[@]} + 1 ))

echo "=== Step 1 of $STEPS · hpcc/share_data.sh: link data/videos and data/dataset to the team's shared folders"
bash "$REPO/hpcc/share_data.sh"

step=1
for pipeline in "${PIPELINES[@]}"; do
    step=$((step + 1))
    echo
    echo "=== Step $step of $STEPS · $pipeline/hpcc/setup_env.sh: build the environment of $pipeline"
    (cd "$REPO/$pipeline" && bash hpcc/setup_env.sh)          # in brackets, so the cd holds for this line only
done
echo
echo "Done. SAM 3 and SAM 3.1 still need Meta's weights: run the two 'hf' lines each of them printed above,"
echo "once, with a Hugging Face account that has been approved for facebook/sam3 and facebook/sam3.1."
echo "Next: LABELLING.md"
