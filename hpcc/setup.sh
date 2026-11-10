#!/bin/bash
# Everything one person sets up once on the MSU HPCC. Run it on a development node, which has internet:
#
#     ssh dev-amd20-v100
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/setup.sh
#
# It links data/videos and data/dataset to the team's folders on the research space (hpcc/share_data.sh) and
# builds three environments: SAM 2.1 (which also runs the YOLO pipelines), SAM 3 and SAM 3.1 (each pipeline's
# hpcc/setup_env.sh). Safe to run again: it only adds what is missing.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$REPO/hpcc/share_data.sh"
for pipeline in sam2_pipeline sam3_pipeline sam3_1_pipeline; do
    echo
    echo "=== $pipeline/hpcc/setup_env.sh"
    (cd "$REPO/$pipeline" && bash hpcc/setup_env.sh)
done
echo
echo "Done. SAM 3 and SAM 3.1 still need Meta's weights: run the two 'hf' lines each of them printed above,"
echo "once, with a Hugging Face account that has been approved for facebook/sam3 and facebook/sam3.1."
echo "Next: LABELLING.md"
