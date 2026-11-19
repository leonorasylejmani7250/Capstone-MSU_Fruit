# Step 2: the detector that finds clusters, and
# the detector → SAM 2.1 pipeline built on it.
DETECTOR = F.UltralyticsModel("../yolo26_pipeline/model/best.pt", conf=0.25, imgsz=512)
# Zero-shot instead (downloads the weights):
# DETECTOR = F.UltralyticsModel("yoloe-26m-seg.pt", text="grape cluster", conf=0.2)

# The fine-tuned SAM 2.1 to test: the newest one the team ships in model/ (see
# model/README.md), else Meta's pretrained weights. Set it to another .pt file to test a
# different checkpoint (training writes them to data/models/).
_shipped = sorted(Path("model").glob("sam2.1_*.pt"), key=lambda p: p.stat().st_mtime)
FINETUNED = _shipped[-1] if _shipped else None

PIPELINE = F.DetectThenSAM2(DETECTOR, ckpt=FINETUNED)
# loads the detector and SAM 2.1 now, so model loading is not timed as inference
PIPELINE.prepare()
print(
    "detector:",
    Path(DETECTOR.weights).name,
    "· SAM 2.1:",
    FINETUNED.name if FINETUNED else f"{F.SAM2_SIZE} pretrained",
)
