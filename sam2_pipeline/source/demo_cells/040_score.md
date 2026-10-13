## 3. Score pretrained SAM 2.1
SAM 2 does not find grapes by itself; it segments and **follows** what it is pointed at. Here it is given one box per cluster, on the first frame where the cluster is well in view, and must track it through the clip. That is the job it does in the full pipeline, where a detector supplies the boxes.

**J&F** is the standard video-segmentation score: the average of **J** (mask overlap, IoU) and **F** (how well the outlines agree), from 0 to 100, over the frames after the hint.
