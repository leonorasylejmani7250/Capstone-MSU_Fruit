#@title Step 4 · Score the starting line-up on the val clips, before any training  { display-mode: "form" }
#@markdown Add or remove lines in `LINEUP` to compare whatever you like: other SAM 2.1 sizes, your own YOLO
#@markdown weights, predictions written by another notebook (`FromFolder`), SAM 3 (see Appendix B).
#@markdown
#@markdown The *automatic* rows use a YOLOE detector steered by a text prompt. Wording matters: in the team's earlier
#@markdown YOLO test "green grapes" scored higher than "grape cluster" on the sample clip. Try both; the table decides.
INCLUDE_AUTOMATIC = True          #@param {type:"boolean"}
DETECTOR_WEIGHTS  = "yoloe-26m-seg.pt"   #@param {type:"string"}
DETECTOR_TEXT     = "grape cluster"      #@param {type:"string"}
DETECTOR_CONF     = 0.2           #@param {type:"slider", min:0.05, max:0.8, step:0.05}

LINEUP = {
    f"SAM 2.1 {SAM2_SIZE} · pretrained · box":   SAM2Prompted(prompt="box"),      # hint = a box, like a detector gives
    f"SAM 2.1 {SAM2_SIZE} · pretrained · click": SAM2Prompted(prompt="point"),    # hint = one click
    # "SAM 2.1 large · pretrained · box":        SAM2Prompted("large", prompt="box"),
    # "my YOLO-seg":                             UltralyticsModel("/path/to/best.pt"),
    # "from another notebook":                   FromFolder("/path/to/predictions", tracks=True),
}
if INCLUDE_AUTOMATIC:
    DETECTOR = UltralyticsModel(DETECTOR_WEIGHTS, text=DETECTOR_TEXT or None, conf=DETECTOR_CONF)
    LINEUP[f"{Path(DETECTOR_WEIGHTS).stem} '{DETECTOR_TEXT}'"] = DETECTOR
    LINEUP[f"{Path(DETECTOR_WEIGHTS).stem} '{DETECTOR_TEXT}' → SAM 2.1 {SAM2_SIZE} pretrained"] = DetectThenSAM2(DETECTOR)

board = benchmark(LINEUP)
