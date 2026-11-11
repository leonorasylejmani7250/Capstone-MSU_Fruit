## Count every photo
The pipeline reads *clips*: a folder of frames named `00000.jpg`, `00001.jpg`, … (that is also how Meta's SAM 2 reads video). A photo becomes a one-frame clip. For each photo the cell records the detector's box count, the number of objects in SAM 2.1's mask, and saves the mask drawn on the photo to `data/count_test/results/overlays/`.
