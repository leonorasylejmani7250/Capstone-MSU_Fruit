# Scalable Fruit Yield Estimation Using Video-Based Computer Vision

CMSE 495 (Michigan State University) capstone with National Grape and MSU AgBioResearch: a computer-vision pipeline that finds, outlines, follows and counts grape clusters in GoPro, phone and drone video of vineyards.

## Run it on the MSU HPCC

Every block can be pasted into a terminal as it is. [`hpcc/README.md`](hpcc/README.md) explains each step.

**Log in.** Use your NetID in place of `netid`. Paste the two lines one at a time: `ssh` swallows whatever is pasted after it.

```bash
ssh netid@hpcc.msu.edu
```

```bash
ssh dev-amd24-h200
```

**1. Set up, once per person.**

```bash
git clone --filter=blob:none https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit
bash hpcc/setup.sh
```

The first line copies this repository into your HPCC home folder; `--filter=blob:none` leaves the old versions of files on GitHub, which keeps the download small. `hpcc/setup.sh` then gets your HPCC account ready for the pipelines, in 30 to 60 minutes. It runs on the HPCC only: nothing is installed or changed on your own computer. It does two things:

- **Shared data**: `data/videos`, `data/dataset` and `data/wgisd` become links to the folders of the same names in `/mnt/research/MSU-Fruit`, so the team has one copy of the video, one set of clips and labels and one copy of the public WGISD dataset, which is fetched if it is not there yet. Videos or clips already in the clone move there first. Everything else in `data/` stays in the clone: each person's own runs and results. You have to be a member of the `MSU-Fruit` research space: if you are not, the script stops here and nothing has been changed.
- **Environments**: `~/envs/fruit_sam2` for SAM 2.1, which also runs the YOLO pipelines, then `~/envs/fruit_sam3` and `~/envs/fruit_sam3_1`. An environment is a folder with its own Python and packages, apart from the HPCC's software. With the first one come Meta's SAM 2 code and the pretrained weights, in `~/.fruit_sam2`, and a Jupyter kernel, so that a notebook can run in that environment.

All of it takes about 25 GB of the 100 GB in your home folder. The script needs no administrator rights, changes no login file such as `~/.bashrc`, submits no jobs and asks no questions. [`hpcc/README.md`](hpcc/README.md#what-setupsh-does) lists every folder it makes, where each download comes from and how to remove it all, and says [what each environment file does and how to change it](hpcc/README.md#the-environments).

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

**2. Bring in the video:** [`data/README.md`](data/README.md).

**3. Label clips:** [`LABELLING.md`](LABELLING.md).

**4. Fine-tune every model and score them all.**

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/run_all.sh grapes_v1
```

**5. Count the clusters in every video, with every model.**

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/count_rows.sh
```

Steps 4 and 5 submit batch jobs and return at once; `squeue --me` lists the jobs until they are done. Two results come out, both in `data/results/`: `leaderboard_val.csv` (how good each model is on our labelled clips) and `cluster_counts_by_video.csv` (each model's cluster total for each video). [`RESULTS.md`](RESULTS.md) has the leaderboard of the latest training round, with every row and column explained.

## The pipelines

One folder per model, one shared `data/` folder. Each folder's README has that pipeline's own steps for the HPCC, as blocks to paste.

| Pipeline | What it does | On a laptop |
|---|---|---|
| **SAM 2.1** — [`sam2_pipeline/`](sam2_pipeline/README.md) | The labelling screen, fine-tuning of SAM 2.1, the leaderboard of all models, and the video pipeline: a YOLO detector finds the clusters, SAM 2.1 outlines and follows them. | [`INSTALL.md`](sam2_pipeline/INSTALL.md), or Google Colab |
| **SAM 3** — [`sam3_pipeline/`](sam3_pipeline/README.md) | Finds and follows clusters from the text prompt "grape cluster", as released (zero-shot) and fine-tuned on our clips. It can also follow the clusters a YOLO detector finds. | needs an NVIDIA GPU |
| **SAM 3.1** — [`sam3_1_pipeline/`](sam3_1_pipeline/README.md) | The same with SAM 3.1, Meta's newer release with a faster multi-object tracker. | needs an NVIDIA GPU |
| **YOLO26** — [`yolo26_pipeline/`](yolo26_pipeline/README.md) | Trains YOLO26n-seg on the public WGISD grape photos, scores it, counts clusters in video with tracker ids. | [`INSTALL.md`](yolo26_pipeline/INSTALL.md) |
| **YOLO11** — [`yolo11_pipeline/`](yolo11_pipeline/README.md) | The same scripts with YOLO11n-seg, the model the team shipped in October. | [`INSTALL.md`](yolo11_pipeline/INSTALL.md) |

**Training and fine-tuning.** The YOLO models are *trained* on the public WGISD photos (`train.py` in their folders). On our own labelled clips every model is *fine-tuned* from its starting point: SAM 2.1 from Meta's pretrained weights, YOLO26 and YOLO11 from their WGISD models, SAM 3 and SAM 3.1 from Meta's release. SAM 3 and SAM 3.1 are also scored *zero-shot*, as released. Each leaderboard row says which it is.

## Layout

```
Capstone-MSU_Fruit/
├── LABELLING.md       how to label clips
├── RESULTS.md         the leaderboard of the latest training round, explained
├── data/              everything the pipelines read and write (data/README.md)
│   ├── get_wgisd.py   fetches the public WGISD grape dataset into wgisd/ (not in git)
│   ├── wgisd_yolo/    WGISD as YOLO labels, with the 27 test photos (in git)
│   ├── videos/        the partner's video (not in git; shared by the team on the HPCC)
│   ├── dataset/       our clips and their labels (not in git; shared by the team on the HPCC)
│   ├── predictions/   every model's masks on the clips, what the leaderboard scores
│   └── results/       the leaderboard, the cluster counts, then sam2/ sam3/ sam3_1/ yolo26/ yolo11/
├── reproducibility/   every guide as one file, pinned to a commit (README.md, "Reproducibility")
├── sam2_pipeline/     code, install guide, environment, model/
├── sam3_pipeline/     code, guide, model/
├── sam3_1_pipeline/   the same scripts for SAM 3.1
├── yolo26_pipeline/   code, install guide, model/
├── yolo11_pipeline/   the same scripts with the YOLO11 model
└── hpcc/              setup.sh, run_all.sh, count_rows.sh: the workflow as HPCC batch jobs
```

Three rules: code lives in `<pipeline>/`; everything it reads or writes lives in `data/`; `<pipeline>/model/` holds the model that pipeline ships.

## Reproducibility

`reproducibility/MSU_FRUIT_all_guides.md` is every guide in this repository as one file: the HPCC instructions, the labelling guide, and then each pipeline in turn — SAM 2.1, SAM 3, SAM 3.1, YOLO26, YOLO11. It is generated from the guides on `main`, and its header names the commit it was made from, so a reader can see exactly which version of the instructions they are holding. A link from one guide to another jumps inside the file; links to guides that are not in it (`data/README.md`, the laptop `INSTALL.md` files, the `model/` READMEs) open on GitHub.

It is what to hand in, print, or send to someone who needs the whole method in one document rather than a tour of the repository. Everything in it is also here, in the folder the file came from — when the two disagree, the repository is right and the file is the older copy.

### What a result needs to be reproducible

A number in the leaderboard or the cluster counts can be traced back to the four things that made it: the *video* it came from, the *clips and labels* the model learned from and was scored on, the *model*, and the *code* that ran. Each is kept where it can be named:

| What made the result | Where it lives | How to name it |
|---|---|---|
| The clips and labels, and which clips are `train` and which are `val` | `data/dataset/`, its `clips.json` | The split is fixed once a clip is labelled; a clip never moves between `train` and `val` |
| The model | `data/runs/<pipeline>/<run>/` while training, `<pipeline>/model/best.pt` once shipped | The run name, e.g. `grapes_v1` |
| The predictions a model drew on the val clips | `data/predictions/<model>/` | One folder per model — this is what the leaderboard scores |
| The scores themselves | `data/results/leaderboard_val.csv`, `cluster_counts_by_video.csv` | One row per model; per-cluster detail in `leaderboard_val_details.csv` |
| The code | this repository | The commit hash |

A leaderboard row is only comparable to another if all four match. Two runs that differ in the training clips, or in which clips count as `val`, are two different experiments however similar their numbers look, which is why `data/predictions/` and `data/results/` are kept out of git and regenerated by the scripts rather than committed.

### Running it again

The workflow is `hpcc/README.md`, steps 1 to 5, and each pipeline's README repeats its own step as a block to paste:

```bash
cd ~/Capstone-MSU_Fruit
bash hpcc/setup.sh               # environments and the links to the shared video and clips
bash hpcc/run_all.sh grapes_v1   # fine-tune every model, then score them all on the val clips
bash hpcc/count_rows.sh          # count the clusters in every video, with every model
```

The video and our labelled clips are not in git: they reach a clone through the team's shared folder on the research space [`data/README.md`](data/README.md). A clone without them can build the environments and train on the public WGISD photos, which `python data/get_wgisd.py` fetches, but it cannot reproduce a leaderboard row, because the `val` clips live there.

## Data

Vineyard video (GoPro, phone, DJI) from National Grape, in a SharePoint folder; it reaches the HPCC through Globus [`data/README.md`](data/README.md). Public data: the Embrapa WGISD grape dataset (CC BY-NC 4.0). It is not in git: `python data/get_wgisd.py` fetches it from its own repository into `data/wgisd/` [`data/README.md`](data/README.md). Its 27 test photos are in git, in `data/wgisd_yolo/`.

## Tools

Python · PyTorch · Segment Anything (SAM 2.1, SAM 3, SAM 3.1) · Ultralytics YOLO (YOLO26, YOLO11, YOLOE) · OpenCV · Jupyter / Google Colab · MSU HPCC
