# Install · YOLO26 pipeline

Install, test the install, score the shipped model. No GPU and no partner data needed. Every command runs from this folder, `yolo26_pipeline/`.

## 1. What you need

- Git: <https://git-scm.com/downloads>
- Python 3.10 or newer: <https://www.python.org/downloads/>
- 1 GB of disk for the environment (5 GB on Linux) and 2 GB for the repository; internet on the first run
- macOS, Linux or Windows; an NVIDIA GPU is optional

## 2. Get the code

```bash
git clone https://github.com/leonorasylejmani7250/Capstone-MSU_Fruit.git
cd Capstone-MSU_Fruit/yolo26_pipeline
```

## 3. Install

macOS and Linux (Terminal):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

If PowerShell refuses `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. For a GPU build of PyTorch on Windows, or a smaller CPU-only build on Linux, take the install line from <https://pytorch.org/get-started/locally/> instead of `python -m pip install torch torchvision`.

## 4. Test the install

```bash
python -c "import torch, ultralytics, cv2; print('ok · torch', torch.__version__, '· ultralytics', ultralytics.__version__, '· CUDA', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

The first line prints the versions (`CUDA False` is fine); the second ends with `OK`.

## 5. Score the shipped model

The WGISD dataset (`../data/wgisd/`, `../data/wgisd_yolo/`) and the model (`model/best.pt`) are in the repository, so this runs at once (about a minute each on a CPU):

```bash
python evaluate.py
python count_test.py
```

`evaluate.py` prints box and mask mAP on the 27 held-out photos; `count_test.py` prints the count error per photo and writes `../data/results/yolo26/count_test.csv` and the annotated photos in `../data/results/yolo26/count_test/`.

Training, video, tracking and fine-tuning on our own footage: [README.md](README.md). Partner video goes in `../data/videos/` ([`../data/README.md`](../data/README.md)).

## 6. If something goes wrong

| Problem | Fix |
| --- | --- |
| `python3` / `py` not found | Install Python from python.org and open a new terminal. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `.\.venv\Scripts\python.exe` instead of `python`. |
| pip stops with `ReadTimeoutError` | Run the same command again with `--timeout 120`. |
| `No module named cv2` or `ultralytics` | Activate the environment (section 3). |
| `Missing weights: …/model/best.pt` | Incomplete clone: `git status`. |
| Out of memory | Lower `--batch` (training) or `--imgsz` (video). |
