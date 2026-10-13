#@title Step 3a · Optional: let a detector draft the labels so you only correct them  { display-mode: "form" }
#@markdown A text-prompted detector (YOLOE) proposes boxes, SAM 2.1 turns them into tracked masks. Drafts stay
#@markdown **unused** until you open each clip below and press *mark done*.
DRAFT_LABELS  = False             #@param {type:"boolean"}
DRAFT_WEIGHTS = "yoloe-26m-seg.pt"  #@param {type:"string"}
DRAFT_TEXT    = "grape cluster"   #@param {type:"string"}
DRAFT_CONF    = 0.2               #@param {type:"slider", min:0.05, max:0.8, step:0.05}

if DRAFT_LABELS and not DEMO:
    draft_labels(DetectThenSAM2(UltralyticsModel(DRAFT_WEIGHTS, text=DRAFT_TEXT, conf=DRAFT_CONF), every=4))
