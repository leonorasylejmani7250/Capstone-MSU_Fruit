#@title Step 5b · Optional: train a YOLO detector on the same labels  { display-mode: "form" }
#@markdown SAM 2.1 only follows what it is pointed at, so the automatic pipeline is only as good as the detector that
#@markdown points. A detector trained on your own labels usually finds far more clusters than a text-prompted one.
#@markdown This writes the labelled clips out in YOLO format (same train / val split) and trains an Ultralytics model.
TRAIN_DETECTOR  = False             #@param {type:"boolean"}
DETECTOR_BASE   = "yolo26n-seg.pt"  #@param ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt"]
DETECTOR_EPOCHS = 100               #@param {type:"integer"}
DETECTOR_IMGSZ  = 1280              #@param [640, 960, 1280] {type:"raw"}

MY_DETECTOR = None
if TRAIN_DETECTOR:
    MY_DETECTOR = UltralyticsModel(train_detector(DETECTOR_BASE, DETECTOR_EPOCHS, DETECTOR_IMGSZ, f"yolo_{_safe(RUN_NAME)}"),
                                   conf=0.25)
    print("detector ready:", MY_DETECTOR.weights)
