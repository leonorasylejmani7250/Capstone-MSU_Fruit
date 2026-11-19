#!/bin/bash
# Set up one person's account on the MSU HPCC for the pipelines. Run it once, on a
# development node (those have internet access):
#
#    ssh dev-amd24-h200
#    cd ~/Capstone-MSU_Fruit
#    bash hpcc/setup.sh
#
# It links data/videos and data/dataset to the team's shared folders and builds the
# conda environments, about 22 GB under ~/envs. It needs no administrator rights, edits
# no login file and submits no jobs. Running it again is safe.
#
# hpcc/README.md, "What setup.sh does", lists every step, everything the script leaves
# in the account and how to remove it.

# Stop at the first command that fails.
set -eo pipefail

# The repository folder, wherever this script is run from.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# One environment for each folder named here. The two YOLO pipelines are not in the
# list: they run in the SAM 2.1 environment.
PIPELINES=(sam2_pipeline sam3_pipeline sam3_1_pipeline)
STEPS=$(( ${#PIPELINES[@]} + 1 ))

echo "=== Step 1 of $STEPS · hpcc/share_data.sh: link data/videos and data/dataset to" \
    "the team's shared folders"
bash "$REPO/hpcc/share_data.sh"

step=1
for pipeline in "${PIPELINES[@]}"; do
    step=$((step + 1))
    echo
    echo "=== Step $step of $STEPS · $pipeline/hpcc/setup_env.sh: build the" \
        "environment of $pipeline"

    # In a subshell, so that the cd holds for this line only.
    (cd "$REPO/$pipeline" && bash hpcc/setup_env.sh)
done
echo
echo "Done. SAM 3 and SAM 3.1 still need Meta's weights: run the two 'hf' lines each" \
    "of them printed above,"
echo "once, with a Hugging Face account that has been approved for facebook/sam3 and" \
    "facebook/sam3.1."
echo "Next: LABELLING.md"
