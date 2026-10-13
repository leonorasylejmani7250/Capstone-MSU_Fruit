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
  python fruit.py video videos/row3.mp4 --seconds 30 --weights models/sam2.1_base_plus_grapes_v1.pt
  python fruit.py train-detector                                     YOLO-seg on the same labels
  python fruit.py approve --all-drafts                               accept drafted labels unchecked
  python fruit.py export-yolo                                        labels out as a YOLO dataset

Labelling needs a screen: open notebooks/MSU_FRUIT_SAM2_pipeline.ipynb in Colab or in OnDemand Jupyter.
Options for every command (--project, --size) can go before or after the command name:
  python fruit.py --project data/msu_fruit_project --size tiny status
--project defaults to $PROJECT_DIR (hpcc/config.sh sets it); --size to $SAM2_SIZE, else base_plus.
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALLER = Path(os.environ.get("FRUIT_CALLER_DIR") or os.getcwd())     # hpcc/run.sb runs from the weights folder
os.environ.setdefault("MPLBACKEND", "Agg")


def here(p):
    """A path typed on the command line: absolute, ~/…, or relative to the folder the command was given in."""
    p = Path(os.path.expandvars(str(p))).expanduser()
    return p if p.is_absolute() else CALLER / p


def load(args):
    """Import the pipeline module and point it at the project folder."""
    if not args.project:
        sys.exit("No project folder: pass --project DIR or set PROJECT_DIR (hpcc/config.sh does).")
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
    print(f"\nvideos/: {len(vids)} file(s)")
    if len(vids):
        print(vids.to_string(index=False))
    print("\nclips:")
    F.dataset_report(show_examples=False)
    models = sorted(F.WEIGHTS.glob("*.pt"))
    print(f"\nfine-tuned weights in models/: {', '.join(m.name for m in models) or 'none yet'}")
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
    found = [p for p in sorted(F.VIDEOS.iterdir()) if p.suffix.lower() in F.VIDEO_EXTS]
    if not found:
        sys.exit(f"No videos in {F.VIDEOS}. Copy them there first (see README: 'Getting videos onto the HPCC').")
    for p in found:
        made = F.cut_clips(p, a.per_video, a.frames, a.fps, a.max_side)
        print(f"{p.name}: {len(made)} clip(s)")
    print("split:", F.split_clips(a.val_fraction))
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


def _finetuned_rows(F, size_filter=None):
    rows = {}
    for ck in sorted(F.WEIGHTS.glob("sam2.1_*.pt")):
        info = F.checkpoint_info(ck)
        size = info.get("size", "?")
        if size_filter and size != size_filter:
            continue
        for prompt, label in (("box", "box"), ("point", "click")):
            rows[f"SAM 2.1 {size} · fine-tuned ({ck.stem.split(size + '_', 1)[-1]}) · {label}"] = \
                F.SAM2Prompted(ckpt=ck, prompt=prompt)
    return rows


def cmd_evaluate(F, a):
    models = F.load_lineup(here(a.lineup)) if a.lineup else {}
    if a.finetuned:
        models.update(_finetuned_rows(F))
    if not models:
        sys.exit("Nothing to score: give --lineup FILE and/or train a model first.")
    print("scoring:", *models, sep="\n  · ")
    F.benchmark(models, split=a.split)
    if a.compare:
        clips = F.get_clips(a.split, labelled=True)
        names = [n for n in models if n.endswith("click")][:3] or list(models)[:3]
        for c in clips[:a.compare]:
            F.compare(c, names)


def cmd_video(F, a):
    video = here(a.video)
    if not video.exists():
        video = F._project_path(a.video)          # e.g. videos/row3.mp4 inside the project folder
    if not video.exists():
        sys.exit(f"cannot find {a.video}")
    if a.detector_weights_trained:
        finder = F.UltralyticsModel(F._project_path(a.detector_weights_trained), conf=a.conf)
    else:
        finder = F.UltralyticsModel(a.detector, text=a.text or None, conf=a.conf)
    ckpt = F._project_path(a.weights) if a.weights else None
    pipeline = F.DetectThenSAM2(finder, ckpt=ckpt)
    print("detector:", Path(finder.weights).name, "· SAM 2.1:", ckpt.name if ckpt else f"{F.SAM2_SIZE} pretrained")
    out, counts, tracks = F.run_on_video(pipeline, video, a.start, a.seconds, a.fps)
    print(tracks.head(30).to_string(index=False))


def cmd_train_detector(F, a):
    best = F.train_detector(a.base, a.epochs, a.imgsz, a.name)
    print("detector weights:", best)


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
                            help="project folder (default: $PROJECT_DIR)")
        parser.add_argument("--size", default=D(os.environ.get("SAM2_SIZE", "base_plus")),
                            choices=["tiny", "small", "base_plus", "large"])
        parser.add_argument("--object-name", default=D(os.environ.get("OBJECT_NAME", "grape cluster")))
        parser.add_argument("--image-size", type=int, default=D(None), help="advanced: SAM 2 input size (default 1024)")

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
                   help="do not add every fine-tuned model in models/ automatically")
    s.add_argument("--split", default="val")
    s.add_argument("--compare", type=int, default=1, help="side-by-side videos for this many clips")

    s = sub.add_parser("video", help="run detector → SAM 2.1 over a stretch of video")
    s.add_argument("video")
    s.add_argument("--start", type=float, default=0)
    s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--fps", type=float, default=6)
    s.add_argument("--weights", default=None, help="fine-tuned SAM 2.1 weights (default: pretrained)")
    s.add_argument("--detector", default="yoloe-26m-seg.pt")
    s.add_argument("--text", default="grape cluster")
    s.add_argument("--conf", type=float, default=0.2)
    s.add_argument("--detector-weights-trained", default=None, help="your own YOLO weights instead of YOLOE")

    s = sub.add_parser("train-detector", help="train a YOLO-seg detector on the same labels")
    s.add_argument("--base", default="yolo26n-seg.pt")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--name", default="yolo_detector")

    s = sub.add_parser("approve", help="mark clips as checked (e.g. imported box → SAM masks you trust)")
    s.add_argument("clips", nargs="*")
    s.add_argument("--all-drafts", action="store_true", help="every clip that is still a draft")

    s = sub.add_parser("export-yolo", help="write the labels as a YOLO segmentation dataset")
    s.add_argument("--every", type=int, default=2)

    a = p.parse_args(argv)
    F = load(a)
    {"status": cmd_status, "download": cmd_download, "clips": cmd_clips, "import-coco": cmd_import_coco,
     "import-yolo": cmd_import_yolo, "train": cmd_train, "evaluate": cmd_evaluate, "video": cmd_video,
     "train-detector": cmd_train_detector, "approve": cmd_approve, "export-yolo": cmd_export_yolo}[a.cmd](F, a)


if __name__ == "__main__":
    main()
