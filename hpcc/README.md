# hpcc/ — the workflow on the MSU HPCC

Five steps. Every block can be pasted into a terminal as it is.

## Log in

Use your NetID in place of `netid`. Paste the two lines one at a time: `ssh` swallows whatever is pasted after it.

```bash
ssh netid@hpcc.msu.edu
```

```bash
ssh dev-amd24-h200
```

The first line lands on a gateway node, which runs no software and refuses `sbatch`. The second goes to a development node, where everything below is typed: the prompt then shows `dev-amd24-h200`.

## 1. Set up, once per person

```bash
git clone --filter=blob:none https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit
bash hpcc/setup.sh
```

The first line copies this repository into your HPCC home folder; `--filter=blob:none` leaves the old versions of files on GitHub, which keeps the download small. `hpcc/setup.sh` then gets your HPCC account ready for the pipelines. It runs on the HPCC only: nothing is installed or changed on your own computer. It takes 30 to 60 minutes, prints each step as it starts, and ends with `Next: LABELLING.md`.

It does five things, in this order:

1. **Shared data.** `data/videos`, `data/dataset` and `data/wgisd` become links to the folders of the same names in `/mnt/research/MSU-Fruit`, so the team has one copy of the video, one set of clips and labels and one copy of the public WGISD dataset. Videos or clips already in the clone move there first. Everything else in `data/` stays in the clone: each person's own runs and results. You have to be a member of the `MSU-Fruit` research space: if you are not, the script stops here and nothing has been changed.
2. **The SAM 2.1 environment**, `~/envs/fruit_sam2`, which also runs the YOLO pipelines. With it come Meta's SAM 2 code and the pretrained weights, in `~/.fruit_sam2`, and a Jupyter kernel, so that a notebook can run in this environment.
3. **The SAM 3 environment**, `~/envs/fruit_sam3`.
4. **The SAM 3.1 environment**, `~/envs/fruit_sam3_1`.
5. **The public WGISD dataset.** The first time anyone on the team runs this, the photos are downloaded into the shared folder (420 MB); after that the step only checks them. It also copies the YOLO training photos into your clone (110 MB).

An environment is a folder with its own Python and packages, apart from the HPCC's software. All of it takes about 25 GB of the 100 GB in your home folder. The script needs no administrator rights, changes no login file such as `~/.bashrc`, submits no jobs and asks no questions. [What `setup.sh` does](#what-setupsh-does) lists every folder it makes, where each download comes from and how to remove it all. [The environments](#the-environments) says what each environment file does and how to change it.

**Meta's weights for SAM 3 and SAM 3.1.** Request access at [facebook/sam3](https://huggingface.co/facebook/sam3) and [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) with one Hugging Face account and wait for the two approval emails. Then log in, once. It asks for a token: make one at <https://huggingface.co/settings/tokens> and paste it.

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
hf auth login
```

Then download the two files, 3 to 4 GB each:

```bash
cd ~/Capstone-MSU_Fruit/sam3_pipeline
source hpcc/activate.sh
hf download facebook/sam3 sam3.pt --local-dir model
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir ../sam3_1_pipeline/model
```

Later, to get the team's latest code:

```bash
cd ~/Capstone-MSU_Fruit
git pull
```

## 2. Bring in the video

[`data/README.md`](../data/README.md), "Video from SharePoint".

## 3. Label clips

[`LABELLING.md`](../LABELLING.md). Labels made in Roboflow or CVAT come in with `fruit.py import-coco` / `import-yolo` ([`sam2_pipeline/README.md`](../sam2_pipeline/README.md), "Labels made elsewhere").

## 4. Fine-tune every model and score them all

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1
```

It submits a chain of batch jobs and returns at once. Run it once per name: while that name's jobs are queued or running, a second call stops and prints the line that cancels them. SAM 2.1 is two jobs: a job has 4 hours, and the second goes on where the first stopped if that was not enough. Every model learns from the labelled `train` clips. The last job scores all of them on the `val` clips, next to the models that were not fine-tuned: `data/results/leaderboard_val.csv`. Each row says what the model is: `· pretrained`, `· trained on WGISD`, `· zero-shot`, or `→ fine-tuned on our clips (<name>)`.

| Model | Starts from | Fine-tuned model |
|---|---|---|
| SAM 2.1 | Meta's pretrained `base_plus` | `data/models/sam2.1_base_plus_<name>.pt` |
| YOLO26 | `yolo26_pipeline/model/best.pt`, trained on WGISD | `data/runs/yolo26_<name>/weights/best.pt` |
| YOLO11 | `yolo11_pipeline/model/best.pt`, trained on WGISD | `data/runs/yolo11_<name>/weights/best.pt` |
| SAM 3 | Meta's release, also scored as released | `sam3_pipeline/model/sam3_<name>.pt` |
| SAM 3.1 | Meta's release, also scored as released | `sam3_1_pipeline/model/sam3_1_<name>.pt` |

A run name is trained once. After labelling more clips, use a new name, and give more steps and epochs if you want them (SAM 2.1 steps, YOLO epochs, SAM 3 epochs; the defaults are 3000, 100 and 20):

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v2 5000 150 30
```

**Ship a winner.** Copy it into its pipeline's `model/` folder (a YOLO model over `model/best.pt`) and fill in that folder's README. Step 5 counts with the models in the `model/` folders.

## 5. Count the clusters in every video

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/count_rows.sh
```

One folder under `data/videos`, or one video:

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/count_rows.sh 2026/Grape--DJIOsmo
```

One job per pipeline, each with the model it ships:

| Pipeline | How it counts |
|---|---|
| SAM 2.1 | YOLO26 finds the clusters, SAM 2.1 outlines and follows each one. A cluster keeps one id for as long as it is in view. |
| YOLO26, YOLO11 | Tracker ids (ByteTrack). An id counts once it has been followed for a second. |
| SAM 3, SAM 3.1 | The model's own ids, from the text prompt "grape cluster". |

The totals land in `data/results/`:

- `cluster_counts.csv`: one line per video and model.
- `cluster_counts_by_video.csv`: one row per video, one column per model.
- `sam2/`, `yolo26/`, `yolo11/`, `sam3/`, `sam3_1/`: each pipeline's overlay video and per-frame counts.

A video that has its total is skipped, so run the script again when new video arrives or a job ran out of time. A total is the number of clusters a model followed, not a hand count: count one row by hand and compare before relying on the numbers.

## Watch the jobs

```bash
squeue --me
```

`PD` is waiting, `R` is running, and a job that is no longer listed has finished. Jobs keep running when the terminal is closed or the connection drops. A job that asks for 4 hours or less usually starts within minutes. Stop one with `scancel` and its number from the first column.

The end of the newest log:

```bash
tail -n 20 "$(ls -t ~/Capstone-MSU_Fruit/*_pipeline/*.out | head -1)"
```

Each pipeline's logs are in its own folder, named `<job name>-<job id>.out`: `fruit-…` in `sam2_pipeline`, `sam3-…`, `sam3_1-…`, `yolo26-…`, `yolo11-…` in theirs.

## One pipeline at a time

Each pipeline's README has its own steps as blocks to paste: [SAM 2.1](../sam2_pipeline/README.md), [SAM 3](../sam3_pipeline/README.md), [SAM 3.1](../sam3_1_pipeline/README.md), [YOLO26](../yolo26_pipeline/README.md), [YOLO11](../yolo11_pipeline/README.md). They use two kinds of command:

- **In the terminal**, for what takes a few minutes: `source hpcc/activate.sh` in the pipeline's folder, then the command. A process on a development node gets two hours of CPU time; the activate scripts keep a program to 8 CPU threads so that this lasts.
- **As a batch job**, for training and for whole folders of video: `sbatch hpcc/run.sb …` from the pipeline's folder.

## Getting a better model

- **Labels decide everything.** More labelled clips from more videos (rows, light, cameras) beat more training steps on the same clips.
- **SAM 2.1:** start at 3000 steps; then `--lr`, `--encoder yes` (trains the image backbone too; needs a 40 GB GPU) and `--init` (continue from an earlier run). `python fruit.py train --help`.
- **YOLO on our clips:** `train-detector` trains at 1280 px for 100 epochs; next try `--epochs 200`, or a bigger base (`--base yolo26s-seg.pt`).
- **SAM 3 and SAM 3.1:** `finetune.py --epochs 40`, or `--resolution 768` if the GPU runs out of memory.
- **The video pipeline with both fine-tuned:** [`sam2_pipeline/README.md`](../sam2_pipeline/README.md), step 5. To score that pair on the leaderboard, switch on the commented entry at the end of `sam2_pipeline/configs/lineup.yaml`.

## What `setup.sh` does

`hpcc/setup.sh` runs five scripts, one after the other, and stops at the first one that fails. A download that fails in step 5 can be finished later ([If something goes wrong](#if-something-goes-wrong)). Each script says in its comments what every step does. What to edit to change something is in "The environments" below.

| Step | Script | What it does |
|---|---|---|
| 1 | `hpcc/share_data.sh` | Makes `data/videos`, `data/dataset` and `data/wgisd` links to the team's shared folders. |
| 2 | `sam2_pipeline/hpcc/setup_env.sh` | Builds `~/envs/fruit_sam2`, fetches Meta's SAM 2 code and the pretrained weights, registers the Jupyter kernel, tests the GPU. |
| 3 | `sam3_pipeline/hpcc/setup_env.sh` | Builds `~/envs/fruit_sam3` and checks that Meta's `sam3` package loads. |
| 4 | `sam3_1_pipeline/hpcc/setup_env.sh` | The same for `~/envs/fruit_sam3_1`. |
| 5 | `data/get_wgisd.py` | Fetches the public WGISD dataset into the shared folder when it is not complete there, checks every file, and copies the YOLO training photos into `data/wgisd_yolo/images/`. |

**What it leaves in your account**

| Where | What | Size |
|---|---|---|
| `~/envs/fruit_sam2`, `~/envs/fruit_sam3`, `~/envs/fruit_sam3_1` | The three environments: Python 3.12, PyTorch and each pipeline's packages. | about 22 GB together |
| `~/.fruit_sam2/` | Meta's SAM 2 code and the pretrained SAM 2.1, YOLOE and YOLO26 weights. | about 1 GB |
| `~/.local/share/jupyter/kernels/fruit_sam2/` | The Jupyter kernel "Python (fruit_sam2)". | a few small files |
| `~/.cache/pip` | pip's copy of what it downloaded. Safe to delete. | about 3 GB |
| `~/Capstone-MSU_Fruit/data/` | The links `videos`, `dataset` and `wgisd`, the YOLO training photos in `wgisd_yolo/images/`, and the empty folders `predictions/` and `models/`. | 110 MB |
| `/mnt/research/MSU-Fruit/wgisd` | The public WGISD dataset, in the team's shared folder: fetched by the first person who runs the setup. | 420 MB, once for the team |

conda, pip, Ultralytics and matplotlib also keep a few small settings and cache files under `~/.conda`, `~/.config` and `~/.cache`.

**Where the downloads come from**

| What | From |
|---|---|
| Python 3.12 and pip | conda-forge, through the HPCC's own conda (the Miniforge3 module) |
| PyTorch, the CUDA 12.6 build | `download.pytorch.org` |
| The other Python packages | PyPI. Each pipeline's `requirements.txt` lists them and says what they are for. |
| Meta's SAM 2 code, Meta's `sam3` package, Ultralytics' CLIP | GitHub: `facebookresearch/sam2`, `facebookresearch/sam3`, `ultralytics/CLIP` |
| The pretrained SAM 2.1 weights | Meta's file server, `dl.fbaipublicfiles.com`. If it cannot be reached: the copy in Ultralytics' releases on GitHub. |
| The YOLOE and YOLO26 weights | Ultralytics' releases on GitHub |
| The WGISD dataset | GitHub: `thsant/wgisd`, the files listed in `data/wgisd_files.sha256` |

**What it does not do**

- It installs nothing on your own computer. Everything above is in your HPCC account.
- It needs no administrator rights and asks no questions.
- It changes no login file such as `~/.bashrc`. The modules and the environment a script loads are gone again when that script ends.
- It submits no batch jobs and trains nothing.
- It does not log in to Hugging Face and does not download Meta's SAM 3 weights. Those are the two blocks that follow it in step 1.
- In a fresh clone it moves and deletes nothing. Only when `data/videos`, `data/dataset` or `data/wgisd` already hold files does step 1 copy them to the shared folder and put the link in the folder's place. A file that the shared folder already has under the same name stays as it is there.

**Running it again** is safe: an environment that exists is kept and its packages are checked against the list, a file that is there is not downloaded again, a link that is right is left alone.

**Removing it.** Delete the folders of the first table: `rm -rf ~/envs/fruit_sam2 ~/envs/fruit_sam3 ~/envs/fruit_sam3_1 ~/.fruit_sam2 ~/.local/share/jupyter/kernels/fruit_sam2`. Deleting `~/Capstone-MSU_Fruit` as well removes your clone, with the models and results of your own runs. The shared video, labels and WGISD photos in `/mnt/research/MSU-Fruit` stay.

## The environments

Each pipeline runs in a conda environment: a folder under `~/envs/` with its own Python and packages.

| Environment | Runs | Built by | Its packages |
|---|---|---|---|
| `~/envs/fruit_sam2` | SAM 2.1, YOLO26, YOLO11 and the labelling notebook | `sam2_pipeline/hpcc/setup_env.sh` | `sam2_pipeline/requirements.txt` |
| `~/envs/fruit_sam3` | SAM 3 | `sam3_pipeline/hpcc/setup_env.sh` | `sam3_pipeline/requirements.txt` |
| `~/envs/fruit_sam3_1` | SAM 3.1 | `sam3_1_pipeline/hpcc/setup_env.sh` | `sam3_1_pipeline/requirements.txt` |
| `~/envs/fruit_yolo26`, `~/envs/fruit_yolo11` | YOLO26 or YOLO11 by itself. Optional: `hpcc/setup.sh` does not build them. | `hpcc/setup_env.sh` in `yolo26_pipeline/` or `yolo11_pipeline/` | that folder's `requirements.txt` |

A YOLO pipeline uses the SAM 2.1 environment until its own has been built.

Every pipeline folder has the same environment files, named here as they are inside that folder. Each says in its comments what its steps do.

| File | What it does | When it runs |
|---|---|---|
| `hpcc/setup_env.sh` | Builds the environment: makes the folder with Python 3.12, installs PyTorch, installs `requirements.txt`, checks that the packages load. | Once per person, by `hpcc/setup.sh` or by hand |
| `requirements.txt` | The list of Python packages, with what each one is for. | Read by `setup_env.sh` |
| `hpcc/activate.sh` | Switches a terminal to the environment, so that `python` is the environment's own. It lasts until the terminal is closed. | `source hpcc/activate.sh` at the top of a block |
| `hpcc/run.sb` | Does the same inside a batch job, then runs one command. | `sbatch hpcc/run.sb …` |
| `sam2_pipeline/hpcc/config.sh` | The SAM 2.1 pipeline's paths: the environment, the SAM 2 code and weights, the `data/` folder. | Read by the SAM 2.1 scripts, `hpcc/run_all.sh` and `hpcc/count_rows.sh` |
| `sam2_pipeline/environment.yml` | The SAM 2.1 environment for a laptop ([`INSTALL.md`](../sam2_pipeline/INSTALL.md)). The HPCC does not use it. | `conda env create` |

**Change an environment.** Edit the file, then run that pipeline's `setup_env.sh` again on a development node. It keeps what is there and installs what is new.

| To change | Edit | Then |
|---|---|---|
| A package: add one, or fix its version | The pipeline's `requirements.txt` | Run `setup_env.sh` again. A package taken off the list stays installed until it is uninstalled or the environment is built from scratch. |
| The version of Meta's `sam3` code | The `sam3[train]` line of `requirements.txt`: end it with `@` and a commit of Meta's repository. Leave it as it is to get Meta's newest code. | After `source hpcc/activate.sh`: `python -m pip uninstall -y sam3`. Then run `setup_env.sh` again. Without the uninstall, pip keeps the copy that is installed. |
| The version of Meta's SAM 2 code | `SAM2_COMMIT` in `sam2_pipeline/hpcc/setup_env.sh` and in `sam2_pipeline/source/cells/030_setup.py` | In `sam2_pipeline/`, after `source hpcc/activate.sh`: `python source/build.py`, and commit the files it regenerates. Delete `~/.fruit_sam2/sam2_repo`, then run `setup_env.sh` again. |
| The PyTorch or CUDA build | The `--index-url` line of `setup_env.sh`. `cu126` is the CUDA 12.6 build. | After `source hpcc/activate.sh`: `python -m pip uninstall -y torch torchvision`. Then run `setup_env.sh` again. |
| The Python version | `python=3.12` in `setup_env.sh` | Delete the environment's folder, then run `setup_env.sh` again. |
| Where the SAM 2.1 environment is kept | `ENV_DIR` in `sam2_pipeline/hpcc/config.sh` | Run `setup_env.sh` again, delete the old folder, and give the new path in the Jupyter form ([`LABELLING.md`](../LABELLING.md), step 1). |
| Where another environment is kept | The default path in its `setup_env.sh`, `activate.sh` and `run.sb`, and where `hpcc/run_all.sh` and `hpcc/count_rows.sh` name it | Run `setup_env.sh` again, then delete the old folder. |
| Which environments `hpcc/setup.sh` builds | The `PIPELINES` line of `hpcc/setup.sh` | Nothing more. |
| The team's shared folder | The `SHARED` line of `hpcc/share_data.sh` | In each clone: remove the links `data/videos`, `data/dataset` and `data/wgisd`, then `bash hpcc/share_data.sh`. |
| An environment that no longer works | Nothing | Delete its folder, then run `setup_env.sh` again. |

## If something goes wrong

| What you see | Do this |
|---|---|
| `You are currently using a gateway node`, or `module(s) could not be loaded: "Miniforge3"` | You are on a gateway node. `ssh dev-amd24-h200`, then paste the block again. |
| `-bash: dev-amd24-h200: command not found` | It needs `ssh` in front: `ssh dev-amd24-h200`. |
| `Unable to open file hpcc/run.sb`, or `can't open file …/train.py` | The wrong folder. Every block starts with the `cd` line it needs; paste the whole block. |
| `Killed` in the middle of a run in the terminal | The process used up its two hours of CPU time. Start from `source hpcc/activate.sh` in the pipeline's folder, which limits the CPU threads, or submit the run as a batch job. |
| A job stays `PD` for a long time | Jobs that ask for more than 4 hours wait much longer. The scripts here ask for 4 hours or less; a job that runs out of time continues when it is submitted again. |
| `conda` stops with `No module named …` before anything of ours runs | Check the cluster's own conda: `(module purge; module load Miniforge3; conda --version)`. If that fails too, it is the HPCC, not your setup: try later, or write to ICER through <https://contact.icer.msu.edu>. |
| `WGISD is not complete yet` at the end of the setup, or `No training photos in …` | On a development node, in `sam2_pipeline/` after `source hpcc/activate.sh`: `python ../data/get_wgisd.py`. It goes on where it stopped. |
| `401` or `GatedRepoError` from `hf download` | Access to that model is not approved yet, or `hf auth login` was not done (step 1). |
