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
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git ~/Capstone-MSU_Fruit
cd ~/Capstone-MSU_Fruit
bash hpcc/setup.sh
```

**2. Bring in the video:** [`data/README.md`](data/README.md). **3. Label clips:** [`LABELLING.md`](LABELLING.md).

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

Steps 4 and 5 submit batch jobs and return at once; `squeue --me` lists the jobs until they are done. Two results come out, both in `data/results/`: `leaderboard_val.csv` (how good each model is on our labelled clips) and `cluster_counts_by_video.csv` (each model's cluster total for each video).

## The pipelines

One folder per model, one shared `data/` folder. Each folder's README has that pipeline's own steps for the HPCC, as blocks to paste.

| Pipeline | What it does | On a laptop |
|---|---|---|
| **SAM 2.1** — [`sam2_pipeline/`](sam2_pipeline/README.md) | The labelling screen, fine-tuning of SAM 2.1, the leaderboard of all models, and the video pipeline: a YOLO detector finds the clusters, SAM 2.1 outlines and follows them. | [`INSTALL.md`](sam2_pipeline/INSTALL.md), or Google Colab |
| **SAM 3** — [`sam3_pipeline/`](sam3_pipeline/README.md) | Finds and follows clusters from the text prompt "grape cluster", as released (zero-shot) and fine-tuned on our clips. | needs an NVIDIA GPU |
| **SAM 3.1** — [`sam3_1_pipeline/`](sam3_1_pipeline/README.md) | The same with SAM 3.1, Meta's newer release with a faster multi-object tracker. | needs an NVIDIA GPU |
| **YOLO26** — [`yolo26_pipeline/`](yolo26_pipeline/README.md) | Trains YOLO26n-seg on the public WGISD grape photos, scores it, counts clusters in video with tracker ids. | [`INSTALL.md`](yolo26_pipeline/INSTALL.md) |
| **YOLO11** — [`yolo11_pipeline/`](yolo11_pipeline/README.md) | The same scripts with YOLO11n-seg, the model the team shipped in October. | [`INSTALL.md`](yolo11_pipeline/INSTALL.md) |

**Training and fine-tuning.** The YOLO models are *trained* on the public WGISD photos (`train.py` in their folders). On our own labelled clips every model is *fine-tuned* from its starting point: SAM 2.1 from Meta's pretrained weights, YOLO26 and YOLO11 from their WGISD models, SAM 3 and SAM 3.1 from Meta's release. SAM 3 and SAM 3.1 are also scored *zero-shot*, as released. Each leaderboard row says which it is.

## Layout

```
Capstone-MSU_Fruit/
├── LABELLING.md       how to label clips
├── data/              everything the pipelines read and write (data/README.md)
│   ├── wgisd/         public WGISD grape dataset (in git)
│   ├── wgisd_yolo/    WGISD as YOLO labels (in git)
│   ├── videos/        the partner's video (not in git; shared by the team on the HPCC)
│   ├── dataset/       our clips and their labels (not in git; shared by the team on the HPCC)
│   ├── predictions/   every model's masks on the clips, what the leaderboard scores
│   └── results/       the leaderboard, the cluster counts, then sam2/ sam3/ sam3_1/ yolo26/ yolo11/
├── sam2_pipeline/     code, install guide, environment, model/
├── sam3_pipeline/     code, guide, model/
├── sam3_1_pipeline/   the same scripts for SAM 3.1
├── yolo26_pipeline/   code, install guide, model/
├── yolo11_pipeline/   the same scripts with the YOLO11 model
└── hpcc/              setup.sh, run_all.sh, count_rows.sh: the workflow as HPCC batch jobs
```

Three rules: code lives in `<pipeline>/`; everything it reads or writes lives in `data/`; `<pipeline>/model/` holds the model that pipeline ships.

## Data

Vineyard video (GoPro, phone, DJI) from National Grape, in a SharePoint folder; it reaches the HPCC through Globus ([`data/README.md`](data/README.md)). Public data: the Embrapa WGISD grape dataset (CC BY-NC 4.0), in `data/wgisd/`.

## Tools

Python · PyTorch · Segment Anything (SAM 2.1, SAM 3, SAM 3.1) · Ultralytics YOLO (YOLO26, YOLO11, YOLOE) · OpenCV · Jupyter / Google Colab · MSU HPCC
