#!/bin/bash
# One videos folder and one dataset folder (the clips and their labels) for the whole team, on the research space.
# Each person runs this once, in their own clone (hpcc/setup.sh runs it as its first step):
#
#     cd ~/Capstone-MSU_Fruit
#     bash hpcc/share_data.sh                        shared folder: /mnt/research/MSU-Fruit
#     bash hpcc/share_data.sh /mnt/research/<group>  another research space
#
# It makes <shared>/videos and <shared>/dataset (writable by the group) and links data/videos and data/dataset to
# them. Videos and clips this clone already holds move to the shared folders first. Everything else in data/
# (predictions, models, runs, results) stays in the clone: each person's own runs.
#
# WHAT IT CHANGES, and nothing else
#   in the shared folder    makes videos/ and dataset/ if they are not there yet
#   in this clone           data/videos and data/dataset become links to those two folders
# A fresh clone has no data/videos and no data/dataset, so there nothing is moved and nothing is deleted.
# Where they do hold files, every file is copied to the shared folder and the clone's folder is then
# deleted and replaced by the link. A file that the shared folder already has under the same name is not
# copied: the shared one stays as it is.
#
# WHEN IT STOPS, with a message
#   - the shared folder is not one you can write to (you are not in its group): nothing has been changed
#   - data/videos or data/dataset is already a link to some other folder
#   - the clone and the shared folder both hold labelled clips: two sets of clips are not merged
#
# Safe to run again: a link that is right is left alone.
# TO CHANGE the shared folder for everyone, edit the default in the SHARED= line below.
set -euo pipefail          # stop at the first command that fails, and at a variable that was never set
SHARED="${1:-/mnt/research/MSU-Fruit}"                            # the folder given on the command line, else the team's
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"      # this clone's data/ folder, wherever this is run from
if [ ! -d "$SHARED" ] || [ ! -w "$SHARED" ]; then
    echo "$SHARED is not a folder you can write to. Give the team's research space: bash hpcc/share_data.sh /mnt/research/<group>" >&2
    exit 1
fi
umask 002          # what is made from here on can be changed by the group, not only by you
for name in videos dataset; do
    shared="$SHARED/$name"; mine="$DATA/$name"
    # The shared folder, the first time anyone runs this.
    if [ ! -d "$shared" ]; then
        mkdir "$shared"; chmod 2770 "$shared"          # the group reads and writes; new files keep the group
    fi
    # Already a link: the right one is left alone, any other one is for you to look at.
    if [ -L "$mine" ]; then
        if [ "$(readlink -f "$mine")" = "$(readlink -f "$shared")" ]; then echo "data/$name → $shared (linked already)"; continue; fi
        echo "data/$name is a link to $(readlink "$mine"), not to $shared. Remove the link and run this again." >&2
        exit 1
    fi
    # A real folder in the clone: what it holds goes to the shared folder, then the folder makes way for the link.
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
            chmod -R g+rwX "$shared" 2>/dev/null || true          # the group can read and change what was just copied
        fi
        rm -rf "$mine"
    fi
    ln -s "$shared" "$mine"
    echo "data/$name → $shared"
done
echo "Next: videos into $SHARED/videos (data/README.md), python fruit.py clips, then LABELLING.md."
