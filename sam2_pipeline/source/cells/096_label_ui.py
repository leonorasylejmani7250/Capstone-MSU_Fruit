#@title Step 3b · Label: box each cluster once, SAM 2.1 follows it through the clip  { display-mode: "form" }
#@markdown - **Drag a box** around a cluster on any frame → a new object, tracked forwards and backwards through the clip.
#@markdown - Step through the frames (◀ ▶). A cluster that only comes into view later: go to that frame and box it there.
#@markdown - **Wrong mask on some frame?** Click the object (click mode *Select*), press **Re-box selected**, and drag the
#@markdown   correct box on that frame. This is the reliable fix. *Add to / Remove from selected* clicks are gentle nudges;
#@markdown   SAM often needs a few of them. **Delete selected** removes the object from the whole clip. **Undo** steps back.
#@markdown - When every cluster in every frame is right, press **Clip checked: mark done**, then pick the next clip in the menu.
#@markdown
#@markdown Label **every** cluster you can see in the val clips: a model that finds a real cluster you skipped gets marked wrong.
if DEMO:
    print("Demo mode: the sample clips already have labels. With your own videos the labelling screen opens here.")
else:
    label()
