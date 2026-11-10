#!/usr/bin/env python
"""Command line for the MSU FRUIT SAM 2.1 pipeline. The HPCC job scripts call this; it also works on any
computer with the environment from hpcc/setup_env.sh. It runs the same code as the Colab notebook.

  python fruit.py status                                   what is in the project, what is usable
  python fruit.py download --sizes tiny base_plus          fetch weights now (compute nodes may be offline)
  python fruit.py clips                                    cut every video in videos/ into clips, split train/val
  python fruit.py import-coco export/_annotations.coco.json          labels from Roboflow / CVAT / Label Studio
  python fruit.py import-yolo export/images export/labels            YOLO outlines or boxes
  python fruit.py train --run grapes_v1 --steps 3000                 fine-tune SAM 2.1 (Meta's trainer)
  python fruit.py evaluate --lineup configs/lineup.yaml              score every model on the val clips
  python fruit.py video videos/row3.mp4                              count the clusters in a video: YOLO26 finds them, SAM 2.1 follows them
  python fruit.py counts                                             every model's total for every video
  python fruit.py check videos/row3.mp4                              a model finds nothing: which detector and size see this video
  python fruit.py train-detector --base yolo26_pipeline/model/best.pt   fine-tune our YOLO26 on the same labels
  python fruit.py approve --all-drafts                               accept drafted labels unchecked
  python fruit.py export-yolo                                        labels out as a YOLO dataset

Labelling needs a screen: open notebooks/MSU_FRUIT_SAM2_pipeline.ipynb in Colab or in OnDemand Jupyter.
Options for every command (--project, --size) can go before or after the command name:
  python fruit.py --project ../data/demo --size tiny status
--project defaults to $PROJECT_DIR (hpcc/config.sh sets it), else to the repository's data/ folder;
--size to $SAM2_SIZE, else base_plus. `evaluate` scores the lineup plus every model it finds in the project:
SAM 2.1 weights in model/ (the ones the team ships) and in the project's models/, the detectors that
train-detector wrote to the project's runs/, and the SAM 3 / SAM 3.1 masks in predictions/ (each folder with a
recipe.json). `video` uses the YOLO26 in yolo26_pipeline/model/ and the newest SAM 2.1 weights in model/.
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                              # sam2_pipeline/
DEFAULT_PROJECT = HERE.parent / "data"                                # the repository's shared data folder
MODEL_DIR = HERE / "model"                                            # the fine-tuned SAM 2.1 the team ships
CALLER = Path(os.environ.get("FRUIT_CALLER_DIR") or os.getcwd())     # hpcc/run.sb runs from the weights folder
os.environ.setdefault("MPLBACKEND", "Agg")


def here(p):
    """A path typed on the command line: absolute, ~/…, or relative to the folder the command was given in."""
    p = Path(os.path.expandvars(str(p))).expanduser()
    return p if p.is_absolute() else CALLER / p


def load(args):
    """Import the pipeline module and point it at the project folder."""
    if not args.project:
        args.project = str(DEFAULT_PROJECT)
    sys.path.insert(0, str(HERE))
    import pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    import fruit_sam2 as F
    F.SAM2_SIZE = args.size
    F.OBJECT_NAME = args.object_name
    if args.image_size:
        F.SAM2_IMAGE_SIZE = args.image_size
    root = F.use_project(here(args.project))
    where = f"GPU {F.GPU_NAME} ({F.GPU_GB:.0f} GB)" if F.DEVICE == "cuda" else "CPU (no GPU found)"
    print(f"project {root} · SAM 2.1 {args.size} · {where}", flush=True)
    return F


def cmd_status(F, a):
    vids = F.list_videos()
    print(f"\nvideos/: {len(vids)} file(s) (any folder depth; clips = cut from it, done = labelled and marked done)")
    if len(vids):
        print(vids.to_string(index=False))
    print("\nclips:")
    F.dataset_report(show_examples=False)
    shipped = sorted(MODEL_DIR.glob("*.pt"))
    print(f"\nthe team's fine-tuned SAM 2.1, in sam2_pipeline/model/: {', '.join(m.name for m in shipped) or 'none yet'}")
    models = sorted(F.WEIGHTS.glob("*.pt"))
    print(f"fine-tuned weights in the project's models/: {', '.join(m.name for m in models) or 'none yet'}")
    for f in sorted(F.RESULTS.glob("leaderboard_*.csv")):
        if f.name.endswith("_details.csv"):
            continue
        import pandas as pd
        print(f"\n{f.name}:")
        F.show_board(pd.read_csv(f))


def cmd_download(F, a):
    (F.BASE / "weights").mkdir(parents=True, exist_ok=True)
    os.chdir(F.BASE / "weights")                  # Ultralytics saves here; hpcc/run.sb runs jobs from here
    for size in a.sizes:
        print("SAM 2.1", size, "→", F.sam2_checkpoint(size))
    if a.detector:
        m = F.UltralyticsModel(a.detector, text=a.text or None)
        m.prepare()                                   # downloads the weights (and YOLOE's text encoder)
        print("detector ready:", a.detector)
    for w in a.yolo:
        F._ultralytics().YOLO(w)
        print("YOLO base weights ready:", w)


def cmd_clips(F, a):
    found = F.find_videos()                      # every video under videos/, any depth; files still syncing wait for the next run
    if not found:
        sys.exit(f"No videos under {F.VIDEOS}. Put them there, or sync them from SharePoint (data/README.md).")
    new = 0
    for p in found:
        made = F.cut_clips(p, a.per_video, a.frames, a.fps, a.max_side)
        new += len(made)
        print(f"{F.video_key(p)}: {len(made)} clip(s)")
    print(f"{len(found)} video(s), {new} clip(s)")
    print("split:", F.split_clips(a.val_fraction))
    table = F.list_videos()
    table.to_csv(F.DATA / "videos.csv", index=False)           # what has come in and what has been cut from it
    print(f"video list: {F.DATA / 'videos.csv'}")
    F.dataset_report(show_examples=False)


def cmd_import_coco(F, a):
    F.VAL_FRACTION = a.val_fraction
    F.import_coco(here(a.json), here(a.images) if a.images else None, split=a.split, categories=a.categories,
                  source=a.source)


def cmd_import_yolo(F, a):
    F.VAL_FRACTION = a.val_fraction
    F.import_yolo_seg(here(a.images), here(a.labels), split=a.split, classes=a.classes, source=a.source)


def cmd_train(F, a):
    t0 = time.time()
    init = None
    if a.init:
        init = here(a.init) if here(a.init).exists() else F._project_path(a.init)
    out = F.finetune(a.run, size=a.size, steps=a.steps, lr=a.lr, encoder=a.encoder, amp_dtype=a.amp, init_ckpt=init)
    F.plot_training(a.run)
    print(f"weights: {out}  ({(time.time() - t0) / 60:.1f} min)")
    print("next: python fruit.py evaluate   (fine-tuned weights in models/ are scored automatically)")
    print(f"to ship these weights: copy them to {MODEL_DIR} and update model/README.md")


def _finetuned_rows(F, size_filter=None):
    """Every fine-tuned SAM 2.1: the shipped ones in sam2_pipeline/model/ and the project's own in models/."""
    rows = {}
    seen = set()
    for where, ck in [("shipped", c) for c in sorted(MODEL_DIR.glob("sam2.1_*.pt"))] + \
                     [("", c) for c in sorted(F.WEIGHTS.glob("sam2.1_*.pt"))]:
        if ck.name in seen:
            continue
        seen.add(ck.name)
        info = F.checkpoint_info(ck)
        size = info.get("size", "?")
        if size_filter and size != size_filter:
            continue
        for prompt, label in (("box", "box"), ("point", "click")):
            rows[f"SAM 2.1 {size} · fine-tuned on our clips ({ck.stem.split(size + '_', 1)[-1]}{', ' + where if where else ''}) · {label}"] = \
                F.SAM2Prompted(ckpt=ck, prompt=prompt)
    return rows


def _detector_rows(F):
    """Every detector fine-tuned on the project's labels (train-detector): runs/<name>/weights/best.pt. The row says
    what it started from: e.g. 'yolo26n-seg · WGISD → fine-tuned on our clips (yolo26_grapes_v1)'."""
    import json
    rows = {}
    for best in sorted(F.RUNS.glob("*/weights/best.pt")):
        run = best.parents[1]
        recipe = json.loads((run / "recipe.json").read_text()) if (run / "recipe.json").exists() else {}
        base = Path(recipe.get("base", "")).name
        if base == "best.pt":                                   # one of our shipped models (yolo26_pipeline/model/best.pt …)
            start = f"{Path(recipe['base']).parents[1].name.replace('_pipeline', '')} · WGISD"
        elif base:
            start = base.replace(".pt", "") + " · pretrained"
        else:
            start = "YOLO-seg"
        rows[f"{start} → fine-tuned on our clips ({run.name})"] = F.UltralyticsModel(best, conf=0.25, imgsz=int(recipe.get("imgsz", 1280)))
    return rows


def _folder_rows(F):
    """Masks written into the project's predictions/ by another pipeline, each folder with a recipe.json that names the
    row (sam3_pipeline and sam3_1_pipeline write them: '<model> · text 'grape cluster' · zero-shot' or
    '… → fine-tuned on our clips (<run>)'). Scored like any other model."""
    import json
    rows = {}
    for recipe in sorted(F.PRED.glob("*/recipe.json")):
        row = json.loads(recipe.read_text()).get("row") or recipe.parent.name
        rows[row] = F.FromFolder(recipe.parent, tracks=True)
    return rows


def cmd_evaluate(F, a):
    models = F.load_lineup(here(a.lineup)) if a.lineup else {}
    if a.finetuned:
        models.update(_finetuned_rows(F))
        models.update(_detector_rows(F))
        models.update(_folder_rows(F))
    if not models:
        sys.exit("Nothing to score: give --lineup FILE and/or train a model first.")
    print("scoring:", *models, sep="\n  · ")
    if a.fresh:
        F.clear_leaderboard(a.split)
    F.benchmark(models, split=a.split)
    if a.compare:
        clips = F.get_clips(a.split, labelled=True)
        names = [n for n in models if n.endswith("click")][:3] or list(models)[:3]
        for c in clips[:a.compare]:
            F.compare(c, names)


def _video(F, name):
    """A video typed on the command line: a path, or one inside the project folder (videos/row3.mp4)."""
    video = here(name)
    if not video.exists():
        video = F._project_path(name)
    if not video.exists():
        sys.exit(f"cannot find {name}")
    return video


def _detector(F, weights, text, **args):
    """Detector weights typed on the command line: a file (here, in the project or in the repository) or a bare
    Ultralytics name, which is downloaded. Only YOLOE weights use the text."""
    return F.UltralyticsModel(here(weights) if here(weights).exists() else weights, text=text or None, **args)


def cmd_video(F, a):
    video = _video(F, a.video)
    finder = _detector(F, a.detector_weights_trained or a.detector, a.text, conf=a.conf, imgsz=a.detector_imgsz)
    if a.weights:
        ckpt = here(a.weights) if here(a.weights).exists() else F._project_path(a.weights)
        if not ckpt.exists():
            sys.exit(f"cannot find the SAM 2.1 weights {a.weights} (looked in the current folder and the project)")
    else:                                               # default: the newest fine-tuned SAM 2.1 the team ships
        shipped = sorted(MODEL_DIR.glob("sam2.1_*.pt"), key=lambda c: c.stat().st_mtime)
        ckpt = shipped[-1] if shipped else None
    pipeline = F.DetectThenSAM2(finder, ckpt=ckpt)
    print("SAM 2.1:", ckpt.name if ckpt else f"{F.SAM2_SIZE} pretrained")
    if not video.is_dir():
        out, counts, tracks = F.run_on_video(pipeline, video, a.start, a.seconds, a.fps)
        return
    videos = F.find_videos(video)                       # a folder: every video in it, whole; counted ones are skipped
    done = set() if a.overwrite else F.counted(F.model_label(pipeline))
    for i, v in enumerate(videos, 1):
        if F.video_key(v) in done:
            print(f"[{i}/{len(videos)}] {F.video_key(v)}: counted already, skipping (--overwrite to redo)")
            continue
        print(f"[{i}/{len(videos)}] {F.video_key(v)}", flush=True)
        try:
            F.run_on_video(pipeline, v, fps=a.fps)
        except AssertionError as e:                     # one unreadable file does not stop the folder
            print(f"  skipped: {e}")
    F.count_table()


def cmd_counts(F, a):
    F.count_table()


def cmd_check(F, a):
    video = _video(F, a.video)
    if a.weights:
        found = {Path(w).name if Path(w).name != "best.pt" else "/".join(Path(w).parts[-3:]): _detector(F, w, a.text, conf=a.conf)
                 for w in a.weights}
    else:                                               # every detector the leaderboard scores
        models = F.load_lineup(here(a.lineup)) if a.lineup else {}
        models.update(_detector_rows(F))
        found, seen = {}, set()
        for name, m in models.items():
            d = getattr(m, "detector", m)
            if isinstance(d, F.UltralyticsModel) and (d.weights, str(d.text)) not in seen:
                seen.add((d.weights, str(d.text)))
                found[name.split(" → SAM")[0].split(" · zero-shot")[0]] = d
    if not found:
        sys.exit("No detector to check: give --weights FILE.")
    F.check_detectors(video, found, frames=a.frames, conf=a.conf, max_side=a.max_side)


def cmd_train_detector(F, a):
    base = a.base
    if "/" in base or here(base).exists():          # a file (ours, or a project / repository path), not a bare name to download
        for candidate in (here(base), F._project_path(base)):
            if candidate.exists():
                base = str(candidate); break
    best = F.train_detector(base, a.epochs, a.imgsz, a.name, batch=a.batch)
    print("detector weights:", best)
    print("next: python fruit.py evaluate   (detectors in the project's runs/ are scored automatically)")


def cmd_approve(F, a):
    t = F.clips_table()
    names = list(t.loc[t["labels"] == "draft", "clip"]) if a.all_drafts else a.clips
    if not names:
        sys.exit("nothing to approve: name clips, or pass --all-drafts")
    unknown = sorted(set(names) - set(t["clip"]))
    if unknown:
        sys.exit(f"no such clip(s): {unknown[:5]}")
    F.mark_done(names)
    print(f"marked {len(names)} clip(s) done; they now count for training and scoring")


def cmd_export_yolo(F, a):
    F.export_yolo_seg(every=a.every)


def main(argv=None):
    def common_options(parser, defaults=True):
        D = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
        parser.add_argument("--project", default=D(os.environ.get("PROJECT_DIR")),
                            help="project folder (default: $PROJECT_DIR, else the repository's data/ folder)")
        parser.add_argument("--size", default=D(os.environ.get("SAM2_SIZE", "base_plus")),
                            choices=["tiny", "small", "base_plus", "large"])
        parser.add_argument("--object-name", default=D(os.environ.get("OBJECT_NAME", "grape cluster")))
        parser.add_argument("--image-size", type=int, default=D(None), help="advanced: SAM 2 input size (default 1024)")

    size = lambda v: v if v == "auto" else int(v)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_options(p)
    common = argparse.ArgumentParser(add_help=False)      # the same options also work after the command name
    common_options(common, defaults=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *a, **k: _add(*a, parents=[common], **k)

    sub.add_parser("status", help="what is in the project")

    s = sub.add_parser("download", help="fetch weights now (run on a dev node: compute nodes may be offline)")
    s.add_argument("--sizes", nargs="+", default=["tiny", "base_plus"], choices=["tiny", "small", "base_plus", "large"])
    s.add_argument("--detector", default="yoloe-26m-seg.pt", help="detector weights to fetch ('' for none)")
    s.add_argument("--text", default="grape cluster", help="text prompt (YOLOE also needs its text encoder)")
    s.add_argument("--yolo", nargs="*", default=["yolo26n-seg.pt"], help="base weights for train-detector")

    s = sub.add_parser("clips", help="cut videos into clips and split train/val")
    s.add_argument("--per-video", type=int, default=6)
    s.add_argument("--frames", type=int, default=16)
    s.add_argument("--fps", type=float, default=6)
    s.add_argument("--max-side", type=int, default=1280)
    s.add_argument("--val-fraction", type=float, default=0.25)

    for name, helptext in (("import-coco", "import a COCO segmentation export"), ("import-yolo", "import YOLO labels")):
        s = sub.add_parser(name, help=helptext)
        if name == "import-coco":
            s.add_argument("json")
            s.add_argument("--images", default=None, help="image folder (default: next to the json)")
            s.add_argument("--categories", nargs="*", default=None)
        else:
            s.add_argument("images")
            s.add_argument("labels")
            s.add_argument("--classes", nargs="*", type=int, default=None)
        s.add_argument("--split", default="auto", choices=["auto", "train", "val"])
        s.add_argument("--source", default=None, help="name for this batch of labels (default: folder name)")
        s.add_argument("--val-fraction", type=float, default=0.25)

    s = sub.add_parser("train", help="fine-tune SAM 2.1")
    s.add_argument("--run", default="grapes_v1", help="name of this run; the same name resumes it")
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--lr", type=float, default=5e-6)
    s.add_argument("--encoder", default="auto", choices=["auto", "no", "yes"], help="train the image backbone too?")
    s.add_argument("--amp", default="auto", choices=["auto", "bfloat16", "float16"])
    s.add_argument("--init", default=None, help="start from these SAM 2.1 weights instead of Meta's")

    s = sub.add_parser("evaluate", help="score models on the labelled val clips")
    s.add_argument("--lineup", default=str(HERE / "configs" / "lineup.yaml"), help="YAML list of models ('' for none)")
    s.add_argument("--no-finetuned", dest="finetuned", action="store_false",
                   help="do not add the models found in the project automatically (SAM 2.1 weights in model/ and models/, "
                        "detectors in runs/, SAM 3 / SAM 3.1 masks in predictions/)")
    s.add_argument("--split", default="val")
    s.add_argument("--fresh", action="store_true",
                   help="start the table over: drop the rows kept from earlier runs (renamed or removed models stay otherwise)")
    s.add_argument("--compare", type=int, default=1, help="side-by-side videos for this many clips")

    s = sub.add_parser("video", help="count the clusters in a video: a detector finds them, SAM 2.1 follows them")
    s.add_argument("video", help="a video, or a folder: every video in it")
    s.add_argument("--start", type=float, default=0, help="seconds into the video")
    s.add_argument("--seconds", type=float, default=0, help="how much of it (default 0: to the end; 20 for a quick look)")
    s.add_argument("--fps", type=float, default=6)
    s.add_argument("--overwrite", action="store_true", help="with a folder: count again the videos that have a total")
    s.add_argument("--weights", default=None,
                   help="fine-tuned SAM 2.1 weights (default: the newest in model/, else pretrained)")
    s.add_argument("--detector", default="yolo26_pipeline/model/best.pt",
                   help="the detector that finds the clusters: YOLO weights (default: our YOLO26; runs/<name>/weights/best.pt "
                        "for one fine-tuned on our clips) or yoloe-26m-seg.pt, which goes by --text")
    s.add_argument("--text", default="grape cluster", help="what a YOLOE detector looks for")
    s.add_argument("--conf", type=float, default=0.25, help="keep boxes that score at least this")
    s.add_argument("--detector-weights-trained", default=None, help=argparse.SUPPRESS)        # old name of --detector
    s.add_argument("--detector-imgsz", type=size, default="auto",
                   help="the detector's input size: auto (the size at which it is most sure on this video) or a number")

    sub.add_parser("counts", help="every model's cluster total for every video (results/cluster_counts.csv)")

    s = sub.add_parser("check", help="a model finds nothing: what every detector sees in a video, at every input size")
    s.add_argument("video")
    s.add_argument("--weights", nargs="+", default=None,
                   help="detector weights to check (default: the lineup's detectors and the ones fine-tuned on our clips)")
    s.add_argument("--text", default="grape cluster", help="what a YOLOE detector looks for")
    s.add_argument("--lineup", default=str(HERE / "configs" / "lineup.yaml"))
    s.add_argument("--frames", type=int, default=6, help="frames to look at, spread over the video")
    s.add_argument("--conf", type=float, default=0.25, help="a box counts from this score")
    s.add_argument("--max-side", type=int, default=1280, help="frames are shrunk to this first, as clips and video do (0: full size)")

    s = sub.add_parser("train-detector", help="train a YOLO-seg detector on the same labels")
    s.add_argument("--base", default="yolo26n-seg.pt",
                   help="starting weights: a bare Ultralytics name, or a file such as yolo26_pipeline/model/best.pt")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    s.add_argument("--name", default="yolo_detector")

    s = sub.add_parser("approve", help="mark clips as checked (e.g. imported box → SAM masks you trust)")
    s.add_argument("clips", nargs="*")
    s.add_argument("--all-drafts", action="store_true", help="every clip that is still a draft")

    s = sub.add_parser("export-yolo", help="write the labels as a YOLO segmentation dataset")
    s.add_argument("--every", type=int, default=2)

    a = p.parse_args(argv)
    F = load(a)
    {"status": cmd_status, "download": cmd_download, "clips": cmd_clips, "import-coco": cmd_import_coco,
     "import-yolo": cmd_import_yolo, "train": cmd_train, "evaluate": cmd_evaluate, "video": cmd_video, "counts": cmd_counts, "check": cmd_check,
     "train-detector": cmd_train_detector, "approve": cmd_approve, "export-yolo": cmd_export_yolo}[a.cmd](F, a)


if __name__ == "__main__":
    main()
