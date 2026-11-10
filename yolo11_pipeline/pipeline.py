"""The names and folders of this pipeline, and load(), which opens a weights file and says what it is.
Every script in this folder imports them."""
from pathlib import Path

NAME = 'yolo11'                                     # this pipeline; names its folders under ../data/runs and ../data/results
BASE_MODEL = 'yolo11n-seg.pt'                       # Ultralytics' pretrained weights that train.py starts from
ROOT = Path(__file__).resolve().parent              # this folder
DATA = ROOT.parent / 'data'                         # the shared data folder of the repository
MODEL = ROOT / 'model' / 'best.pt'                  # the model this pipeline ships
WGISD = DATA / 'wgisd_yolo' / 'grape.yaml'          # the public WGISD grape dataset as YOLO labels (../data/README.md)
RUNS = DATA / 'runs' / NAME                         # training runs: <run>/weights/best.pt, results.csv, curves
RESULTS = DATA / 'results' / NAME                   # everything else this pipeline writes


def describe(ckpt, names, label):
    """What a weights file is, from what Ultralytics saved in it: (one line, warnings, the image size it was
    trained at). Two files find no grape cluster however they are run. A base model, which has no grape class,
    raises ValueError; a training run that was only a test (an epoch or two) gets a warning."""
    args, metrics, results = (ckpt.get(k) or {} for k in ('train_args', 'train_metrics', 'train_results'))
    names = [str(n) for n in names]
    epochs = len(results.get('epoch') or []) or args.get('epochs')
    score = metrics.get('metrics/mAP50(M)', metrics.get('metrics/mAP50(B)'))
    ours = len(names) == 1 or any('grape' in n.lower() for n in names)
    line = f'{label}: ' + ' · '.join(x for x in (
        Path(str(args.get('model') or '')).stem or None,
        f"{epochs} epoch(s) at {args.get('imgsz')} px" if epochs and args.get('imgsz') else None,
        f'val mAP50 {score:.2f}' if score is not None and ours else None,
        f'class: {names[0]}' if len(names) == 1 else f'{len(names)} classes') if x)
    warnings = []
    if not ours:
        raise ValueError(f'{label} is a base model: it knows {len(names)} classes ({", ".join(names[:3])}, ...), none of them a '
                         f'grape cluster. Use a trained model ({MODEL.relative_to(ROOT)}) or train one (train.py).')
    if (epochs and epochs < 5) or (score is not None and score < 0.2):
        warnings.append(f'WARNING: {label} is a test run, not a trained model ({epochs or "?"} epoch(s)'
                        + (f', val mAP50 {score:.2f}' if score is not None else '')
                        + f'). It finds little or nothing on any footage. The trained model is {MODEL.relative_to(ROOT)}.')
    size = args.get('imgsz')
    return line, warnings, int(size) if ours and isinstance(size, (int, float)) else None


def load(weights):
    """Open a weights file and print what it is; a base model stops the script. Returns (the model, the image
    size it was trained at or None)."""
    import sys
    from ultralytics import YOLO
    weights = Path(weights).resolve()
    label = weights.relative_to(ROOT.parent) if weights.is_relative_to(ROOT.parent) else weights
    model = YOLO(str(weights))
    try:
        line, warnings, size = describe(getattr(model, 'ckpt', None) or {}, (model.names or {}).values(), label)
    except ValueError as e:
        sys.exit(str(e))
    print('\n'.join([line, *warnings]), flush=True)
    return model, size
