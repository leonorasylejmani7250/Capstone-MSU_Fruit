#@title Step 1 · Add your videos  { display-mode: "form" }
#@markdown Three ways to get videos in. Use whichever is easiest, then re-run this cell to see the list.
#@markdown 1. **Upload button** (this cell, Colab only). Fine for clips up to a few hundred MB.
#@markdown 2. **Drag and drop** into the `videos` folder shown below using Colab's Files panel (left sidebar). Best for big GoPro files.
#@markdown 3. **Copy them into Google Drive** at *My Drive / PROJECT_NAME / videos* from any computer.
UPLOAD_NOW = False  #@param {type:"boolean"}

if UPLOAD_NOW and not DEMO:
    if IN_COLAB:
        from google.colab import files
        _cwd = os.getcwd(); os.chdir(VIDEOS)            # the upload widget saves into the current folder
        try:
            files.upload()
        finally:
            os.chdir(_cwd)
    else:
        print(f"Not in Colab: copy your videos into {VIDEOS}")

print(f"videos folder: {VIDEOS}")
_v = list_videos()
if len(_v):
    display(_v)
elif not DEMO:
    print("No videos yet. Add some (see the three options above) and run this cell again.")
