#@title Demo data: sample grape clips with ready-made labels (only runs when DEMO is on)  { display-mode: "form" }
if DEMO:
    # Photos and hand-drawn cluster masks come from Embrapa WGISD (Santos et al.), CC BY-NC 4.0:
    # https://github.com/thsant/wgisd. Each photo is turned into a short clip by sliding a window
    # across it, so the "video" has exact ground truth to learn from and to grade against.
    make_demo_clips()
    print("split:", split_clips(VAL_FRACTION))
    display(clips_table())
    show_clip(get_clips("val")[0], title="A demo validation clip with its ground-truth labels")
