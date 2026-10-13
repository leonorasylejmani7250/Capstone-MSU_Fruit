#@title Step 6 · Did fine-tuning help? Same clips, same hints  { display-mode: "form" }
_ft = f"SAM 2.1 {SAM2_SIZE} · fine-tuned ({RUN_NAME})"
LINEUP_FT = {f"{_ft} · box": SAM2Prompted(ckpt=FINETUNED, prompt="box"),
             f"{_ft} · click": SAM2Prompted(ckpt=FINETUNED, prompt="point")}
if INCLUDE_AUTOMATIC:
    LINEUP_FT[f"{Path(DETECTOR_WEIGHTS).stem} '{DETECTOR_TEXT}' → {_ft}"] = DetectThenSAM2(DETECTOR, ckpt=FINETUNED)
if globals().get("MY_DETECTOR") is not None:        # trained in Step 5b
    LINEUP_FT["my YOLO"] = MY_DETECTOR
    LINEUP_FT[f"my YOLO → SAM 2.1 {SAM2_SIZE} pretrained"] = DetectThenSAM2(MY_DETECTOR)
    LINEUP_FT[f"my YOLO → {_ft}"] = DetectThenSAM2(MY_DETECTOR, ckpt=FINETUNED)
board = benchmark(LINEUP_FT)

_val = get_clips("val", labelled=True)
if _val:
    compare(_val[0], [f"SAM 2.1 {SAM2_SIZE} · pretrained · click", f"{_ft} · click"])
