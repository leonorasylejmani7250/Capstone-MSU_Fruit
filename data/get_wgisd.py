"""Fetch the public WGISD grape dataset into data/wgisd.

    python data/get_wgisd.py
        download what is missing, check every file, and copy the YOLO training
        photos to data/wgisd_yolo/images/
    python data/get_wgisd.py --check
        only say what is missing or damaged

About 420 MB from https://github.com/thsant/wgisd, at the commit named below, without
its extra/ and original_resolution/ folders. wgisd_files.sha256 lists the files with
their checksums. Running it again fetches only what is missing. On the HPCC,
hpcc/share_data.sh first makes data/wgisd a link to the team's shared folder, so the
dataset is stored once for everyone.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
import stat
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent  # data/
DEST = HERE / "wgisd"  # the dataset; on the HPCC a link to the shared folder
FILES = HERE / "wgisd_files.sha256"  # "checksum  path" per file, as sha256sum prints
YOLO = HERE / "wgisd_yolo"  # the YOLO labels; their photos are dataset photos
# the dataset's latest commit (23 August 2022)
COMMIT = "6910edc5ae3aae8c20062941b1641821f0c30127"
SOURCE = f"https://raw.githubusercontent.com/thsant/wgisd/{COMMIT}/"
STOP = threading.Event()  # set by Ctrl-C: the downloads not yet started are skipped


def certificates() -> ssl.SSLContext:
    """The system's certificates, and certifi's where that package is installed: the
    Python from python.org on macOS comes without certificates of its own.
    """
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return context


CERTIFICATES = certificates()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def listed() -> dict[str, str]:
    """{path inside the dataset: checksum}, from wgisd_files.sha256."""
    files = {}
    for line in FILES.read_text(encoding="utf-8").splitlines():
        checksum, _, name = line.partition("  ")
        if name:
            files[name] = checksum
    return files


def fetch(name: str, checksum: str) -> str | None:
    """Download one file. Returns None, or what went wrong after four tries."""
    target = DEST / name
    part = target.with_name(f".{target.name}.{os.getpid()}.part")
    problem = "stopped"
    for attempt in range(4):
        if STOP.is_set():
            break
        if attempt:
            time.sleep(2**attempt)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            url = SOURCE + urllib.parse.quote(name)
            with urllib.request.urlopen(
                url, timeout=60, context=CERTIFICATES
            ) as response:
                with part.open("wb") as f:
                    shutil.copyfileobj(response, f)
            if sha256(part) == checksum:
                os.replace(part, target)
                return None
            problem = "the download does not match its checksum"
        except OSError as e:
            problem = str(e)
    part.unlink(missing_ok=True)
    return problem


def place_photos() -> int:
    """Copy every labelled photo to wgisd_yolo/images/<split>/. Returns how many."""
    copied = 0
    for label in sorted((YOLO / "labels").glob("*/*.txt")):
        photo = DEST / "data" / f"{label.stem}.jpg"
        target = YOLO / "images" / label.parent.name / photo.name
        if not photo.is_file():
            continue
        if not target.is_file() or target.stat().st_size != photo.stat().st_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(photo, target)
            copied += 1
    return copied


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--check", action="store_true", help="only say what is missing or damaged"
    )
    args = p.parse_args()

    # In the team's shared folder, what is written stays open to the group.
    if DEST.is_dir() and DEST.stat().st_mode & stat.S_ISGID:
        os.umask(0o002)

    files = listed()
    todo = {
        name: checksum
        for name, checksum in files.items()
        if not (DEST / name).is_file() or sha256(DEST / name) != checksum
    }
    print(f"{DEST}: {len(files) - len(todo)} of {len(files)} files are here and intact")
    if args.check:
        for name in sorted(todo)[:10]:
            print(f"  missing or damaged: {name}")
        sys.exit(
            f"{len(todo)} file(s) to fetch: run this without --check" if todo else 0
        )

    failed = {}
    if todo:
        print(f"Fetching {len(todo)} file(s) from {SOURCE}")
        pool = ThreadPoolExecutor(max_workers=8)
        try:
            results = pool.map(lambda name: fetch(name, todo[name]), todo)
            for done, (name, problem) in enumerate(zip(todo, results), 1):
                if problem:
                    failed[name] = problem
                if done % 100 == 0 or done == len(todo):
                    print(f"  {done} of {len(todo)}", flush=True)
        except KeyboardInterrupt:
            STOP.set()
            sys.exit("Stopped. Run it again to go on where it stopped.")
        finally:
            pool.shutdown()
    if failed:
        name, problem = next(iter(failed.items()))
        hint = "Run it again: it goes on where it stopped."
        if "CERTIFICATE_VERIFY_FAILED" in problem:
            hint = (
                "This Python has no certificates. Run it with the Python of a "
                "pipeline's environment (INSTALL.md)."
            )
        sys.exit(
            f"{len(failed)} file(s) could not be fetched (first: {name}: {problem}).\n"
            + hint
        )

    copied = place_photos()
    counts = ", ".join(
        f"{split} {len(list((YOLO / 'images' / split).glob('*.jpg')))}"
        for split in ("train", "val", "test")
    )
    print(f"WGISD is complete in {DEST}")
    print(f"YOLO photos in {YOLO / 'images'}: {counts} ({copied} copied now)")


if __name__ == "__main__":
    main()
