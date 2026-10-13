#@title Step 7 · Use it: run on a stretch of new video, get an overlay and a count  { display-mode: "form" }
#@markdown Leave `VIDEO_FILE` empty to use the first video in the videos folder.
VIDEO_FILE    = ""     #@param {type:"string"}
START_SECOND  = 0      #@param {type:"number"}
SECONDS       = 10     #@param {type:"number"}
RUN_FPS       = 6      #@param {type:"number"}
USE_FINETUNED = True   #@param {type:"boolean"}

_videos = sorted(p for p in VIDEOS.iterdir() if p.suffix.lower() in VIDEO_EXTS)
_video = VIDEOS / VIDEO_FILE if VIDEO_FILE else (_videos[0] if _videos else None)
if _video is None:
    print("No video to run on yet.")
else:
    if "DETECTOR" not in globals():
        DETECTOR = UltralyticsModel(DETECTOR_WEIGHTS, text=DETECTOR_TEXT or None, conf=DETECTOR_CONF)
    _finder = globals().get("MY_DETECTOR") or DETECTOR          # your trained detector if there is one
    print("detector:", Path(_finder.weights).name, "· SAM 2.1:", "fine-tuned" if USE_FINETUNED and "FINETUNED" in globals() else "pretrained")
    _pipeline = DetectThenSAM2(_finder, ckpt=FINETUNED if USE_FINETUNED and "FINETUNED" in globals() else None)
    _out, counts, tracks = run_on_video(_pipeline, _video, START_SECOND, SECONDS, RUN_FPS)
    display(Video(str(_out), embed=True, html_attributes="controls loop"))
    display(tracks.head(20))
