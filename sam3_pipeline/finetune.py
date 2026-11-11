"""Fine-tune the model on our labelled clips with Meta's trainer, and write a checkpoint that run.py picks up.

    python finetune.py --run grapes_v1               a GPU job on the HPCC: sbatch hpcc/run.sb finetune.py --run grapes_v1

What it does:
  1. our labelled clips (../data/dataset, from the SAM 2.1 labeller) → a COCO dataset in ../data/sam3_dataset/
     (train clips → train/, val clips → test/, one annotation per cluster with its outline)
  2. Meta's trainer (sam3.train.train, started through meta_trainer.py, which corrects one layer so that it can be
     trained) fine-tunes the detector, the part of the model that finds clusters from the text prompt, starting from
     the released weights; log and checkpoint in ../data/runs/<pipeline>_<run>/
  3. the fine-tuned detector is merged into the released video checkpoint → model/<pipeline>_<run>.pt (+ .json recipe)
Then `python run.py clips` uses the new weights (the newest .pt in model/) and `python fruit.py evaluate` in
../sam2_pipeline scores them as "<model> · text '…' → fine-tuned on our clips (<run>)".

--dry-run does step 1, writes the recipe and prints the training command without running it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import pipeline as P

MIN_AREA = 64            # ignore specks smaller than this many pixels


def read_label(path: Path) -> np.ndarray | None:
    """A label PNG as written by the SAM 2.1 labeller: pixel value = cluster number (palette or 16-bit image)."""
    if not path.exists():
        return None
    im = Image.open(path)
    if im.mode not in ('P', 'L', '1', 'I;16', 'I'):
        raise ValueError(f'{path}: expected a PNG whose pixel values are object ids, got a {im.mode} image')
    return np.asarray(im).astype(np.uint16)


def polygons(mask: np.ndarray) -> list[list[float]]:
    """A binary mask → COCO polygons (one per connected part, at least 3 points each)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) >= 3 and cv2.contourArea(c) >= MIN_AREA:
            polys.append([float(v) for v in c.reshape(-1)])
    return polys


def export_coco(data: Path, out_root: Path, text: str, every: int) -> dict:
    """Labelled clips → <out_root>/train and <out_root>/test (our val clips), each with _annotations.coco.json."""
    clips = json.loads((data / 'dataset' / 'clips.json').read_text())
    counts = {}
    for split, folder in (('train', 'train'), ('val', 'test')):
        done = [c for c, info in clips.items() if info.get('split') == split and info.get('status') == 'done']
        out = out_root / folder
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        images, annotations = [], []
        for clip in done:
            frames = sorted((data / 'dataset' / 'JPEGImages' / clip).glob('*.jpg'))
            for t, frame in enumerate(frames):
                if t % every:
                    continue
                label = read_label(data / 'dataset' / 'Annotations' / clip / (frame.stem + '.png'))
                if label is None:
                    continue
                h, w = label.shape[:2]
                name = f'{clip}_{frame.stem}.jpg'
                shutil.copy(frame, out / name)
                image_id = len(images) + 1
                images.append({'id': image_id, 'file_name': name, 'width': int(w), 'height': int(h)})
                for k in np.unique(label):
                    if k == 0:
                        continue
                    m = label == k
                    area = int(m.sum())
                    polys = polygons(m)
                    if area < MIN_AREA or not polys:
                        continue
                    ys, xs = np.nonzero(m)
                    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                    annotations.append({'id': len(annotations) + 1, 'image_id': image_id, 'category_id': 1,
                                        'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': area, 'segmentation': polys,
                                        'iscrowd': 0})
        coco = {'info': {'description': f'MSU FRUIT labelled clips, {split} split, every {every}. frame',
                         'date_created': time.strftime('%Y-%m-%d')},
                'images': images, 'annotations': annotations,
                'categories': [{'id': 1, 'name': text, 'supercategory': text}]}
        (out / '_annotations.coco.json').write_text(json.dumps(coco))
        counts[folder] = {'clips': len(done), 'images': len(images), 'clusters': len(annotations)}
        print(f'{folder}: {len(done)} clips → {len(images)} images, {len(annotations)} cluster outlines → {out}')
    if counts['train']['images'] == 0 or counts['test']['images'] == 0:
        sys.exit('Need labelled clips marked done in both splits (python fruit.py status in ../sam2_pipeline).')
    return counts


def base_checkpoint() -> Path:
    """The released weights: model/<file> if downloaded there, else the Hugging Face cache (needs access and hf auth login)."""
    local = P.MODEL_DIR / P.RELEASE_FILE
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download
    print(f'downloading {P.HF_REPO}/{P.RELEASE_FILE} (once) …')
    return Path(hf_hub_download(repo_id=P.HF_REPO, filename=P.RELEASE_FILE))


def write_config(run: str, dataset_root: Path, log_dir: Path, base: Path, args) -> str:
    """Fill finetune.yaml and put it where Meta's trainer looks for configs (inside the sam3 package)."""
    import sam3.train
    import sam3
    bpe = Path(sam3.__file__).parent / 'assets' / 'bpe_simple_vocab_16e6.txt.gz'
    text = (P.ROOT / 'finetune.yaml').read_text()
    for key, value in {'__DATASET_ROOT__': dataset_root, '__LOG_DIR__': log_dir, '__BPE_PATH__': bpe,
                       '__BASE_CHECKPOINT__': base, '__RESOLUTION__': args.resolution, '__WORKERS__': args.workers,
                       '__EPOCHS__': args.epochs, '__EPOCH_SIZE__': args.epoch_size, '__BATCH__': args.batch}.items():
        text = text.replace(key, str(value))
    name = f'msu_fruit_{P.NAME}_{run}.yaml'
    target = Path(sam3.train.__file__).parent / name
    target.write_text(text)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / name).write_text(text)                      # a copy next to the log, for the record
    return name


def merge(base: Path, trained: Path, out: Path) -> tuple[int, list[str]]:
    """Put the fine-tuned detector weights into the released video checkpoint (detector + tracker)."""
    import torch
    ckpt = torch.load(base, map_location='cpu', weights_only=True)
    state = ckpt['model'] if 'model' in ckpt and isinstance(ckpt['model'], dict) else ckpt
    ft = torch.load(trained, map_location='cpu', weights_only=True)
    ft = ft['model'] if 'model' in ft and isinstance(ft['model'], dict) else ft
    replaced, unmatched = 0, []
    for k, v in ft.items():
        key = f'detector.{k}'
        if key in state and tuple(state[key].shape) == tuple(v.shape):
            state[key] = v
            replaced += 1
        else:
            unmatched.append(k)
    if replaced == 0:
        sys.exit(f'No detector weights matched between {trained} and {base}; nothing written. The trained checkpoint is kept.')
    torch.save({'model': state}, out)
    return replaced, unmatched


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='name of this fine-tuning run, e.g. grapes_v1')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--epoch-size', type=int, default=1500, help='images sampled per epoch')
    p.add_argument('--resolution', type=int, default=1008, help="the model's input size; lower it if the GPU runs out of memory")
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--workers', type=int, default=4, help='data-loading processes')
    p.add_argument('--every', type=int, default=2, help='use every n-th frame of a clip (neighbours are near-copies)')
    p.add_argument('--text', default='grape cluster', help='the category name = the text prompt the model learns')
    p.add_argument('--data', type=Path, default=None, help="project folder (default: $PROJECT_DIR, else the repository's data/)")
    p.add_argument('--dry-run', action='store_true', help='export the dataset and write the recipe, do not train')
    p.add_argument('--merge-only', type=Path, metavar='CHECKPOINT',
                   help='skip training: merge this trained checkpoint (…/checkpoints/checkpoint.pt) into the released weights')
    args = p.parse_args()
    data = (args.data.expanduser().resolve() if args.data else P.DATA)
    dataset_root = data / 'sam3_dataset'
    log_dir = data / 'runs' / f'{P.NAME}_{args.run}'
    out = P.MODEL_DIR / f'{P.NAME}_{args.run}.pt'
    print(f'{P.MODEL_NAME} fine-tuning "{args.run}" · project folder {data}')

    try:
        import sam3.train  # noqa: F401
    except ImportError as e:
        sys.exit(f"Meta's trainer is not installed in this environment ({e}). Install: python -m pip install -r requirements.txt")

    if not args.merge_only:
        counts = export_coco(data, dataset_root, args.text, args.every)
        base = base_checkpoint()
        config = write_config(args.run, dataset_root, log_dir, base, args)
        cmd = [sys.executable, str(P.ROOT / 'meta_trainer.py'), '-c', config, '--use-cluster', '0', '--num-gpus', '1']
        print('training command:', ' '.join(cmd))
        if args.dry_run:
            print(f'dry run: dataset in {dataset_root}, recipe in {log_dir}; nothing trained.')
            return
        subprocess.run(cmd, check=True)
        trained = log_dir / 'checkpoints' / 'checkpoint.pt'
        if not trained.is_file():
            sys.exit(f'training finished but {trained} is missing; see {log_dir}/logs')
    else:
        counts = None
        base = base_checkpoint()
        trained = args.merge_only

    replaced, unmatched = merge(base, trained, out)
    print(f'merged {replaced} detector tensors into {base.name} → {out}' + (f' ({len(unmatched)} not matched)' if unmatched else ''))
    out.with_suffix('.json').write_text(json.dumps({
        'run': args.run, 'pipeline': P.NAME, 'model': P.MODEL_NAME, 'base': str(base), 'trained_detector': str(trained),
        'dataset': str(dataset_root), 'images': counts, 'epochs': args.epochs, 'epoch_size': args.epoch_size,
        'resolution': args.resolution, 'text': args.text, 'unmatched_tensors': unmatched[:20],
        'written': time.strftime('%Y-%m-%d %H:%M')}, indent=1))
    print(f'Next: python run.py clips   (uses {out.name}),  then  cd ../sam2_pipeline && python fruit.py evaluate')


if __name__ == '__main__':
    main()
