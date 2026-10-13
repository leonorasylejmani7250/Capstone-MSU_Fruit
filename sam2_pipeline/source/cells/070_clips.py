#@title Step 2 · Cut the videos into short clips and split them train / val  { display-mode: "form" }
#@markdown A *clip* is a short run of frames (default 16 frames at 6 fps ≈ 2.7 s). You label clips,
#@markdown SAM 2 trains on clips, and every model is scored on clips it has never seen.
CLIPS_PER_VIDEO = 6      #@param {type:"slider", min:1, max:40, step:1}
CLIP_FRAMES     = 16     #@param {type:"slider", min:8, max:48, step:1}
CLIP_FPS        = 6      #@param {type:"number"}
MAX_SIDE        = 1280   #@param [960, 1280, 1920] {type:"raw"}
VAL_FRACTION    = 0.25   #@param {type:"slider", min:0.1, max:0.5, step:0.05}

if not DEMO:
    for _p in sorted(VIDEOS.iterdir()):
        if _p.suffix.lower() in VIDEO_EXTS:
            _made = cut_clips(_p, CLIPS_PER_VIDEO, CLIP_FRAMES, CLIP_FPS, MAX_SIDE)
            print(f"{_p.name}: {len(_made)} clip(s)")
    if len(clips_table()):
        print("split:", split_clips(VAL_FRACTION))
        display(clips_table())
