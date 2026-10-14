#@title Tools: scoring (one yardstick for every model)  { display-mode: "form" }
# Two kinds of model, two leaderboards:
#   prompted  – the model is shown each object once (a box, a click or a mask) and must follow it.
#               Score: J&F, the standard video-segmentation metric (J = overlap, F = edge accuracy).
#   automatic – the model gets only pixels and must find every object itself.
#               Score: precision / recall / F1 at IoU 0.5, mask quality, and how far off the count is.
from scipy.optimize import linear_sum_assignment
from scipy import ndimage

def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)

def _boundary(seg):
    """1-pixel-wide outline of a binary mask (same definition as DAVIS and Meta's SA-V evaluator)."""
    e, s, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]; s[:-1, :] = seg[1:, :]; se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]; b[:, -1] = seg[:, -1] ^ s[:, -1]; b[-1, -1] = False
    return b

def boundary_f(pred, gt, tolerance=0.008):
    """Boundary F-measure: do the two outlines agree to within 0.8% of the image diagonal?"""
    H, W = gt.shape
    r = int(np.ceil(tolerance * np.linalg.norm((H, W))))
    both = pred | gt
    if not both.any():
        return 1.0
    rows, cols = np.flatnonzero(both.any(1)), np.flatnonzero(both.any(0))       # crop: same answer, far faster
    y0, y1 = max(rows[0] - r - 2, 0), min(rows[-1] + r + 3, H)
    x0, x1 = max(cols[0] - r - 2, 0), min(cols[-1] + r + 3, W)
    pb, gb = _boundary(pred[y0:y1, x0:x1]), _boundary(gt[y0:y1, x0:x1])
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy <= r * r).astype(np.uint8)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 or n_g == 0:
        return 1.0 if n_p == n_g else 0.0
    precision = (pb & (cv2.dilate(gb.astype(np.uint8), disk) > 0)).sum() / n_p
    recall = (gb & (cv2.dilate(pb.astype(np.uint8), disk) > 0)).sum() / n_g
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))

def score_prompted(gt, pred, prompts):
    """One row per object, judged on the frames after its prompt.
    J, F          – averaged over frames where the object is in view (in the labels or in the prediction).
                    This is the headline: in a pan most objects leave the frame, and agreeing that an object
                    is gone is easy.
    J_all, F_all  – averaged over every frame, an empty-vs-empty frame counting as perfect. This is the
                    published DAVIS / SA-V convention, so it is comparable with papers but reads higher.
    iou_prompt_frame – mask quality on the very frame the model was pointed at the object."""
    T, rows = gt.shape[0], []
    for pr in prompts:
        k, f0 = pr["obj"], pr["frame"]
        J, F, seen = [], [], []
        for t in range(f0 + 1, T):
            g, p = gt[t] == k, pred[t] == k
            J.append(mask_iou(p, g)); F.append(boundary_f(p, g)); seen.append(bool(g.any() or p.any()))
        J, F, seen = np.array(J, float), np.array(F, float), np.array(seen, bool)
        avg = lambda v: 100 * float(v.mean()) if len(v) else np.nan
        rows.append(dict(obj=k, prompt_frame=f0, frames=len(J), frames_in_view=int(seen.sum()),
                         J=avg(J[seen]), F=avg(F[seen]), J_all=avg(J), F_all=avg(F),
                         iou_prompt_frame=100 * mask_iou(pred[f0] == k, gt[f0] == k)))
    return rows

def iou_matrix(a, b):
    """IoU between every object in id map `a` and every object in id map `b` (same frame).
    Returns (ids_a, ids_b, iou [len(ids_a), len(ids_b)], areas_a, areas_b); background is left out."""
    a_ids, a_inv = np.unique(a, return_inverse=True)
    b_ids, b_inv = np.unique(b, return_inverse=True)
    joint = np.bincount(a_inv.ravel() * len(b_ids) + b_inv.ravel(),
                        minlength=len(a_ids) * len(b_ids)).reshape(len(a_ids), len(b_ids))
    a_area, b_area = joint.sum(1), joint.sum(0)
    ka, kb = a_ids != 0, b_ids != 0
    inter = joint[ka][:, kb]
    iou = inter / np.maximum(a_area[ka][:, None] + b_area[kb][None, :] - inter, 1)
    return a_ids[ka], b_ids[kb], iou, a_area[ka], b_area[kb]

def match_instances(gt, pred, iou_thr=0.5, min_area=0):
    """One frame, no hints: pair predicted objects with labelled ones (best one-to-one pairing by IoU).
    Returns (true positives, false positives, misses, [IoU of each true positive], n_gt, n_pred)."""
    _, _, iou, g_area, p_area = iou_matrix(gt, pred)
    iou = iou[g_area >= min_area][:, p_area >= min_area]
    n_g, n_p = iou.shape
    matched = []
    if n_g and n_p:
        r, c = linear_sum_assignment(-iou)
        matched = [float(v) for v in iou[r, c] if v >= iou_thr]
    return len(matched), n_p - len(matched), n_g - len(matched), matched, n_g, n_p

def score_automatic(gt, pred, iou_thr=0.5, min_area=0):
    """One row per frame for a model that had to find the objects by itself."""
    rows = []
    for t in range(gt.shape[0]):
        tp, fp, fn, ious, n_g, n_p = match_instances(gt[t], pred[t], iou_thr, min_area)
        rows.append(dict(frame=t, tp=tp, fp=fp, fn=fn, iou_sum=sum(ious), n_gt=n_g, n_pred=n_p))
    return rows
