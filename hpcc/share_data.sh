#!/bin/bash
# One videos folder and one dataset folder (the clips and their labels) for the whole team, on the research space.
# Each person runs this once, in their own clone:
#
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/share_data.sh                        shared folder: /mnt/research/MSU-Fruit
#     bash hpcc/share_data.sh /mnt/research/<group>  another research space
#
# It makes <shared>/videos and <shared>/dataset (writable by the group) and links data/videos and data/dataset to
# them. Videos and clips this clone already holds move to the shared folders first. Everything else in data/
# (predictions, models, runs, results) stays in the clone: each person's own runs.
set -euo pipefail
SHARED="${1:-/mnt/research/MSU-Fruit}"
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
if [ ! -d "$SHARED" ] || [ ! -w "$SHARED" ]; then
    echo "$SHARED is not a folder you can write to. Give the team's research space: bash hpcc/share_data.sh /mnt/research/<group>" >&2
    exit 1
fi
umask 002
for name in videos dataset; do
    shared="$SHARED/$name"; mine="$DATA/$name"
    if [ ! -d "$shared" ]; then
        mkdir "$shared"; chmod 2770 "$shared"          # the group reads and writes; new files keep the group
    fi
    if [ -L "$mine" ]; then
        if [ "$(readlink -f "$mine")" = "$(readlink -f "$shared")" ]; then echo "data/$name → $shared (linked already)"; continue; fi
        echo "data/$name is a link to $(readlink "$mine"), not to $shared. Remove the link and run this again." >&2
        exit 1
    fi
    if [ -d "$mine" ]; then
        if [ -n "$(find "$mine" -type f -print -quit)" ]; then
            if [ "$name" = dataset ] && [ -e "$shared/clips.json" ]; then
                echo "data/dataset holds clips, and so does $shared: two sets of clips are not merged here." >&2
                echo "Move yours out of the clone (mv data/dataset ~/dataset_mine) and run this again." >&2
                exit 1
            fi
            echo "moving what data/$name holds to $shared ..."
            (cd "$mine" && find . -type f -print0) | while IFS= read -r -d '' f; do      # a file the shared folder has stays as it is
                [ -e "$shared/$f" ] || { mkdir -p "$shared/$(dirname "$f")"; cp -p "$mine/$f" "$shared/$f"; }
            done
            chmod -R g+rwX "$shared" 2>/dev/null || true
        fi
        rm -rf "$mine"
    fi
    ln -s "$shared" "$mine"
    echo "data/$name → $shared"
done
echo "Next: videos into $SHARED/videos (data/README.md), python fruit.py clips, then LABELLING.md."
