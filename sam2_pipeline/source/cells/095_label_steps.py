#@title Step 3a · Optional: let a detector draft the labels so you only correct them  { display-mode: "form" }
#@markdown A detector proposes boxes, SAM 2.1 turns them into tracked masks. `DRAFT_WEIGHTS`: YOLO weights (our YOLO26
#@markdown by default; outside the repository, YOLOE) or `yoloe-26m-seg.pt`, which looks for `DRAFT_TEXT`. Drafts stay
#@markdown **unused** until you open each clip below, add what the detector missed and press *mark done*.
DRAFT_LABELS  = False             #@param {type:"boolean"}
DRAFT_WEIGHTS = "yolo26_pipeline/model/best.pt"  #@param {type:"string"}
DRAFT_TEXT    = "grape cluster"   #@param {type:"string"}
DRAFT_CONF    = 0.25              #@param {type:"slider", min:0.05, max:0.8, step:0.05}

if DRAFT_LABELS and not DEMO:
    _w = DRAFT_WEIGHTS if "/" not in DRAFT_WEIGHTS or _project_path(DRAFT_WEIGHTS).exists() else "yoloe-26m-seg.pt"
    draft_labels(DetectThenSAM2(UltralyticsModel(_w, text=DRAFT_TEXT, conf=DRAFT_CONF), every=4))
