## Appendix D · When something goes wrong

| What you see | What to do |
|---|---|
| "No GPU" under cell B | Runtime ▸ Change runtime type ▸ T4 GPU, then Run all. |
| Step 4 stops with "No labelled 'val' clips yet" | Expected until clips are labelled **and marked done**. Step 3c lists what is missing. |
| The labelling screen is blank or does not react | Runtime ▸ Restart session, Run all. Still stuck: label from code with `ruler("clip_name", frame=0)` to read box corners and `ed = ClipLabels("clip_name"); ed.add_box(0, [x0, y0, x1, y1]); ed.save(done=True)`, or label in Roboflow / CVAT and import (Appendix A). |
| A mask is wrong on some frames | Select the object, **Re-box selected**, drag the right box on that frame. |
| "CUDA out of memory" while scoring | Pick a smaller `SAM2_SIZE` in cell A, or set `SAM2_IMAGE_SIZE = 768` in the SAM 2.1 tools cell (applies to training and scoring alike, so re-train after changing it). |
| Training is very slow | Check cell B found a GPU. Lower `TRAIN_STEPS`. Keep `TRAIN_ENCODER` on "auto" or "no". |
| Colab disconnected | Run all again. Videos, clips, labels and finished weights are on Drive; an interrupted training run resumes from its last save. |
| Fine-tuned model scores worse | See the reading guide under Step 6. Change `RUN_NAME` before trying new settings so results stay comparable. |
| A video will not open | Convert it: `!ffmpeg -i in.mov -c:v libx264 -crf 18 out.mp4`, and put `out.mp4` in the videos folder. |
