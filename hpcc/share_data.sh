#!/bin/bash
# Give the whole team one videos folder and one dataset folder (the clips and their
# labels) on the research space. Each person runs it once in their own clone;
# hpcc/setup.sh runs it as its first step.
#
#    cd ~/Capstone-MSU_Fruit
#    bash hpcc/share_data.sh
#    bash hpcc/share_data.sh /mnt/research/<group>
#
# Without an argument the shared folder is /mnt/research/MSU-Fruit. The script makes
# <shared>/videos and <shared>/dataset, writable by the group, and turns data/videos and
# data/dataset of this clone into links to them. Files the clone already holds there are
# copied to the shared folder first; a file the shared folder already has is kept.
# Everything else in data/ stays in the clone.
#
# It stops without changing anything when the shared folder is not writable for you,
# when data/videos or data/dataset is already a link to another folder, or when the
# clone and the shared folder both hold labelled clips. Running it again is safe.

# Stop at the first command that fails, and at a variable that was never set.
set -euo pipefail

# The folder given on the command line, else the team's. To change it for everyone, edit
# this default.
SHARED="${1:-/mnt/research/MSU-Fruit}"

# This clone's data/ folder, wherever this script is run from.
DATA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
if [ ! -d "$SHARED" ] || [ ! -w "$SHARED" ]; then
    echo "$SHARED is not a folder you can write to. Give the team's research space:" \
        "bash hpcc/share_data.sh /mnt/research/<group>" >&2
    exit 1
fi

# What is made from here on can be changed by the group, not only by you.
umask 002
for name in videos dataset; do
    shared="$SHARED/$name"
    mine="$DATA/$name"

    # The shared folder, the first time anyone runs this. Mode 2770: the group reads and
    # writes, and new files keep the group.
    if [ ! -d "$shared" ]; then
        mkdir "$shared"
        chmod 2770 "$shared"
    fi

    # Already a link: the right one is left alone, any other one is for you to look at.
    if [ -L "$mine" ]; then
        if [ "$(readlink -f "$mine")" = "$(readlink -f "$shared")" ]; then
            echo "data/$name → $shared (linked already)"
            continue
        fi
        echo "data/$name is a link to $(readlink "$mine"), not to $shared. Remove the" \
            "link and run this again." >&2
        exit 1
    fi

    # A real folder in the clone: what it holds goes to the shared folder, then the
    # folder makes way for the link.
    if [ -d "$mine" ]; then
        if [ -n "$(find "$mine" -type f -print -quit)" ]; then
            if [ "$name" = dataset ] && [ -e "$shared/clips.json" ]; then
                echo "data/dataset holds clips, and so does $shared: two sets of" \
                    "clips are not merged here." >&2
                echo "Move yours out of the clone (mv data/dataset ~/dataset_mine)" \
                    "and run this again." >&2
                exit 1
            fi
            echo "moving what data/$name holds to $shared ..."

            # Copy every file that the shared folder does not have yet.
            (cd "$mine" && find . -type f -print0) | while IFS= read -r -d '' f; do
                [ -e "$shared/$f" ] || {
                    mkdir -p "$shared/$(dirname "$f")"
                    cp -p "$mine/$f" "$shared/$f"
                }
            done

            # The group can read and change what was just copied.
            chmod -R g+rwX "$shared" 2>/dev/null || true
        fi
        rm -rf "$mine"
    fi
    ln -s "$shared" "$mine"
    echo "data/$name → $shared"
done
echo "Next: videos into $SHARED/videos (data/README.md), python fruit.py clips, then" \
    "LABELLING.md."
