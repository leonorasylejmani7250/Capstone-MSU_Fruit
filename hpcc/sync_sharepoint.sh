#!/bin/bash --login
# Pull the team's SharePoint video library into data/videos with rclone, then cut clips from whatever is new.
# (Globus does the same without a script: data/README.md, "Video from SharePoint".)
#
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/sync_sharepoint.sh "sharepoint:Shared Documents/Vineyard video"      the library (or folder) to pull
#     bash hpcc/sync_sharepoint.sh                                                   the remote saved in FRUIT_REMOTE
#
# Once, on a development node (the login opens a browser there; ICER: use ssh -X):
#     module load rclone
#     rclone config          →  new remote "sharepoint", type onedrive, auto config y, then "search" for the team's site
#     export FRUIT_REMOTE="sharepoint:Shared Documents/Vineyard video"     (put it in ~/.bashrc to keep it)
#
# Only videos come over (any folder structure, kept as it is), nothing is ever deleted locally, files still being
# uploaded to SharePoint (changed in the last 2 minutes) wait for the next run, and a run can be repeated any time.
set -eo pipefail
REMOTE="${1:-${FRUIT_REMOTE:-}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/data/videos"
if [ -z "$REMOTE" ]; then
    echo "Which SharePoint folder? bash hpcc/sync_sharepoint.sh \"<remote>:<folder>\", or export FRUIT_REMOTE=... (see the top of this file)" >&2
    exit 1
fi
module load rclone 2>/dev/null || true
if ! command -v rclone >/dev/null; then
    echo "rclone is not available: module load rclone on the HPCC, or install it (https://rclone.org)" >&2
    exit 1
fi
mkdir -p "$DEST"
echo "SharePoint $REMOTE  →  $DEST"
rclone copy "$REMOTE" "$DEST" \
    --include "*.{mp4,mov,m4v,avi,mkv,mts,webm}" --ignore-case \
    --min-age 2m --transfers 4 --checkers 8 --retries 5 --low-level-retries 20 \
    --stats 60s --stats-one-line -v --log-file "$DEST/.sync.log"
echo "synced; log: $DEST/.sync.log"
echo "videos under data/videos: $(find -L "$DEST" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mts' -o -iname '*.webm' \) ! -name '.*' | wc -l)"

cd "$REPO/sam2_pipeline"
source hpcc/activate.sh
python fruit.py clips
echo "Next: label the new clips (sam2_pipeline/README.md, section 2), then bash hpcc/run_all.sh"
