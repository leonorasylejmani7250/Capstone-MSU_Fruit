#@title Tools: labelling  { display-mode: "form" }
import copy, zlib, gc
import anywidget, traitlets, ipywidgets as widgets
if IN_COLAB:
    from google.colab import output as _colab_output
    _colab_output.enable_custom_widget_manager()         # lets Colab show the drawing canvas


def _load_prompts(clip):
    f = PROMPTS / f"{clip}.json"
    return json.loads(f.read_text()) if f.exists() else []

def _save_prompts(clip, prompts):
    PROMPTS.mkdir(exist_ok=True)
    (PROMPTS / f"{clip}.json").write_text(json.dumps(prompts))

class ClipLabels:
    """The labels of one clip plus the operations to change them. No screen needed, so it also works from code:
        ed = ClipLabels("my_clip"); ed.add_box(0, [x0, y0, x1, y1]); ed.delete(3); ed.save(done=True)"""

    def __init__(self, clip, size=None, ckpt=None):
        self.clip = Clip(clip) if isinstance(clip, str) else clip
        self.T = len(self.clip); self.H, self.W = self.clip.hw
        lab = self.clip.labels()
        self.ids = lab if lab is not None else np.zeros((self.T, self.H, self.W), np.uint16)
        self.best = np.where(self.ids > 0, 1000, 0).astype(np.float16)   # existing labels keep their pixels
        self.prompts = _load_prompts(self.clip.name)
        self._model, self._sess, self._undo = (size, ckpt), None, []
        self._on_disk = self.ids.copy() if lab is not None else None       # what the label files currently hold
        self.status_now = (_load_meta().get(self.clip.name, {}).get("status") or "draft") if lab is not None else None

    @property
    def sess(self):
        if self._sess is None:
            self._sess = Sam2Session(load_sam2(self._model[0], self._model[1]), self.clip)
        return self._sess

    def objects(self):
        return sorted(set(np.unique(self.ids).tolist()) - {0})

    def _snapshot(self):
        self._undo = (self._undo + [(zlib.compress(self.ids.tobytes(), 1), zlib.compress(self.best.tobytes(), 1),
                                     copy.deepcopy(self.prompts))])[-20:]

    def _erase(self, obj):
        m = self.ids == obj
        self.ids[m] = 0; self.best[m] = 0

    def _retrack(self, obj):
        self._erase(obj)
        mine = [p for p in self.prompts if p["obj"] == obj]
        if mine:
            self.sess.add(mine, self.ids, self.best, backward=True)
        self.save()

    def _clean_box(self, box):
        x0, y0, x1, y1 = [float(v) for v in box]
        x0, x1 = sorted((min(max(x0, 0), self.W - 1), min(max(x1, 0), self.W - 1)))
        y0, y1 = sorted((min(max(y0, 0), self.H - 1), min(max(y1, 0), self.H - 1)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError("that box is too small")
        return [x0, y0, x1, y1]

    def add_box(self, frame, box):
        """Draw a box around one object on one frame; SAM 2 fills in the mask on every frame. Returns the new id."""
        box = self._clean_box(box)
        new = max([0] + self.objects() + [p["obj"] for p in self.prompts]) + 1
        if new > 255:
            raise ValueError("255 objects per clip is the limit of the label format")
        self._snapshot()
        self.prompts.append(dict(obj=new, frame=int(frame), box=box))
        self._retrack(new)
        return new

    def set_box(self, obj, frame, box):
        """The reliable fix: say where an existing object is on this frame by drawing its box again. Replaces
        any earlier box or clicks for that object on that frame; the other frames are re-tracked from it."""
        obj, frame, box = int(obj), int(frame), self._clean_box(box)
        if obj not in self.objects() and not any(p["obj"] == obj for p in self.prompts):
            raise ValueError(f"there is no object {obj}")
        self._snapshot()          # (an imported or drafted object is re-made from this box on every frame)
        self.prompts = [p for p in self.prompts if not (p["obj"] == obj and p["frame"] == frame)]
        self.prompts.append(dict(obj=obj, frame=frame, box=box))
        self._retrack(obj)

    def add_click(self, obj, frame, x, y, positive=True):
        """Fix an object on one frame: a positive click says 'this belongs to it', a negative click 'this does
        not'. That frame is then re-segmented from the object's outline there plus all clicks on it, and the
        other frames are re-tracked."""
        obj, frame, x, y = int(obj), int(frame), float(x), float(y)
        if not any(p["obj"] == obj for p in self.prompts):
            raise ValueError(f"object {obj} was imported or drafted without a box, so clicks cannot nudge it: "
                             "use Re-box selected (set_box) instead")
        rec = next((p for p in self.prompts if p["obj"] == obj and p["frame"] == frame), None)
        ys, xs = np.nonzero(self.ids[frame] == obj)
        if rec is None and len(xs) == 0 and not positive:
            raise ValueError("the object is not on this frame, so there is nothing to remove")
        self._snapshot()
        if rec is None:                       # start from where the object currently is on this frame
            rec = dict(obj=obj, frame=frame, points=[], labels=[])
            if len(xs): rec["box"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            self.prompts.append(rec)
        rec.setdefault("points", []).append([x, y]); rec.setdefault("labels", []).append(1 if positive else 0)
        if positive and rec.get("box"):       # a 'belongs' click outside the box stretches the box to reach it
            b, pad = rec["box"], 8.0
            rec["box"] = [max(min(b[0], x - pad), 0.0), max(min(b[1], y - pad), 0.0),
                          min(max(b[2], x + pad), self.W - 1.0), min(max(b[3], y + pad), self.H - 1.0)]
        self._retrack(obj)

    def delete(self, obj):
        self._snapshot()
        self._erase(obj)
        self.prompts = [p for p in self.prompts if p["obj"] != obj]
        self.save()

    def undo(self):
        if not self._undo:
            return False
        ids, best, self.prompts = self._undo.pop()
        self.ids = np.frombuffer(zlib.decompress(ids), np.uint16).reshape(self.ids.shape).copy()
        self.best = np.frombuffer(zlib.decompress(best), np.float16).reshape(self.best.shape).copy()
        self.save()
        return True

    def save(self, done=None):
        """Labels are written after every change (only the frames that changed). done=True marks the clip as
        checked by a person; until then it is a draft and is not used for training or scoring."""
        changed = None if self._on_disk is None else {t for t in range(self.T) if not np.array_equal(self.ids[t], self._on_disk[t])}
        self.clip.save_labels(self.ids, only=changed); self._on_disk = self.ids.copy()
        _save_prompts(self.clip.name, self.prompts)
        if done is not None or self.status_now is None:
            self.status_now = "done" if done else "draft"
            set_meta(self.clip.name, status=self.status_now)

    def render(self, frame, selected=None, width=1100):
        """The frame with labels, the boxes/clicks placed on it, and the selected object outlined in white."""
        out = overlay(self.clip.image(frame), self.ids[frame])
        if selected:
            cs, _ = cv2.findContours((self.ids[frame] == selected).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, (0, 0, 0), 5); cv2.drawContours(out, cs, -1, (255, 255, 255), 2)
        for p in self.prompts:
            if p["frame"] != frame: continue
            if p.get("box"):
                x0, y0, x1, y1 = [int(v) for v in p["box"]]
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)
            for (x, y), l in zip(p.get("points") or [], p.get("labels") or []):
                cv2.circle(out, (int(x), int(y)), 6, (0, 0, 0), -1)
                cv2.circle(out, (int(x), int(y)), 4, (60, 255, 60) if l else (255, 60, 60), -1)
        scale = min(1.0, width / self.W)
        if scale < 1:
            out = cv2.resize(out, (round(self.W * scale), round(self.H * scale)), interpolation=cv2.INTER_AREA)
        return out, scale

class _Canvas(anywidget.AnyWidget):
    """Shows an image and reports mouse gestures: a drag (box) or a click, in image pixels."""
    _esm = """
    function render({ model, el }) {
      const wrap = document.createElement("div");
      wrap.style.cssText = "position:relative;display:inline-block;max-width:100%;line-height:0;cursor:crosshair;touch-action:none;user-select:none";
      const img = document.createElement("img");
      img.draggable = false;
      img.style.cssText = "max-width:100%;height:auto;display:block;-webkit-user-drag:none";
      const rect = document.createElement("div");
      rect.style.cssText = "position:absolute;border:2px dashed #fff;box-shadow:0 0 0 1px #000;display:none;pointer-events:none";
      wrap.appendChild(img); wrap.appendChild(rect); el.appendChild(wrap);
      const show = () => { const src = model.get("image"); if (src) img.src = src; };
      model.on("change:image", show); show();
      let start = null;
      const at = (e) => {
        const r = img.getBoundingClientRect();
        const cx = Math.min(Math.max(e.clientX - r.left, 0), r.width), cy = Math.min(Math.max(e.clientY - r.top, 0), r.height);
        return { cx, cy, x: cx * img.naturalWidth / r.width, y: cy * img.naturalHeight / r.height };
      };
      wrap.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault(); start = at(e);
        try { wrap.setPointerCapture(e.pointerId); } catch (err) {}
      });
      wrap.addEventListener("pointermove", (e) => {
        if (!start) return;
        const p = at(e);
        rect.style.left = Math.min(start.cx, p.cx) + "px"; rect.style.top = Math.min(start.cy, p.cy) + "px";
        rect.style.width = Math.abs(p.cx - start.cx) + "px"; rect.style.height = Math.abs(p.cy - start.cy) + "px";
        rect.style.display = "block";
      });
      wrap.addEventListener("pointerup", (e) => {
        if (!start) return;
        const p = at(e), s = start; start = null; rect.style.display = "none";
        const drag = Math.abs(p.cx - s.cx) > 6 && Math.abs(p.cy - s.cy) > 6;
        const ev = drag ? { kind: "box", x0: Math.min(s.x, p.x), y0: Math.min(s.y, p.y), x1: Math.max(s.x, p.x), y1: Math.max(s.y, p.y) }
                        : { kind: "click", x: p.x, y: p.y };
        ev.stamp = Date.now() + Math.random();
        model.set("event", ev); model.save_changes();
      });
      wrap.addEventListener("pointercancel", () => { start = null; rect.style.display = "none"; });
    }
    export default { render };
    """
    image = traitlets.Unicode("").tag(sync=True)
    event = traitlets.Dict({}).tag(sync=True)

class LabelEditor(ClipLabels):
    """ClipLabels with a point-and-click screen on top."""

    def __init__(self, clip, size=None, ckpt=None, on_status=None):
        super().__init__(clip, size, ckpt)
        self.selected, self._scale, self._on_status = None, 1.0, on_status
        self.canvas = _Canvas()
        self.slider = widgets.IntSlider(value=0, min=0, max=max(self.T - 1, 0), description="frame", continuous_update=False,
                                  readout=False, layout=widgets.Layout(width="420px"))
        self.prev, self.next = widgets.Button(description="◀", layout=widgets.Layout(width="40px")), widgets.Button(description="▶", layout=widgets.Layout(width="40px"))
        self.mode = widgets.ToggleButtons(options=[("Select", "select"), ("Add to selected", "pos"), ("Remove from selected", "neg")],
                                    value="select", description="a click will:", style={"button_width": "190px"})
        self.b_del, self.b_undo = widgets.Button(description="Delete selected", icon="trash"), widgets.Button(description="Undo", icon="undo")
        self.b_rebox = widgets.Button(description="Re-box selected", icon="crop", tooltip="then drag the correct box for the selected object on this frame")
        self._rebox = False
        self.b_done = widgets.Button(description="Clip checked: mark done", button_style="success", icon="check",
                               layout=widgets.Layout(width="230px"))
        self.status = widgets.HTML()
        self.slider.observe(lambda c: self.refresh(), names="value")
        self.prev.on_click(lambda b: setattr(self.slider, "value", max(self.slider.value - 1, 0)))
        self.next.on_click(lambda b: setattr(self.slider, "value", min(self.slider.value + 1, self.T - 1)))
        self.b_del.on_click(lambda b: self._do(lambda: self.delete(self.selected) if self.selected else None, clear=True))
        self.b_undo.on_click(lambda b: self._do(self.undo, clear=True))
        self.b_done.on_click(lambda b: self._do(self._approve))
        self.b_rebox.on_click(self._arm_rebox)
        self.canvas.observe(self._on_event, names="event")
        self.view = widgets.VBox([widgets.HBox([self.prev, self.slider, self.next, self.b_done]),
                            widgets.HBox([self.mode, self.b_rebox, self.b_del, self.b_undo]), self.status, self.canvas])
        self.refresh()

    def refresh(self, note=""):
        t = self.slider.value
        img, self._scale = self.render(t, self.selected)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        self.canvas.image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        state = self.status_now or "no labels yet"
        sel = f" · selected <b>#{self.selected}</b>" if self.selected else ""
        self.status.value = (f"<b>{self.clip.name}</b> · frame {t + 1}/{self.T} · {len(self.objects())} object(s) · "
                             f"labels: <b>{state}</b>{sel}" + (f" · {note}" if note else ""))

    def _approve(self):
        self.save(done=True)
        if self._on_status: self._on_status()

    def close(self):
        """Release the clip's frames from the GPU when another clip is opened."""
        self._sess = None
        self.view.close(); gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    def _arm_rebox(self, b=None):
        self._rebox = bool(self.selected)
        self.refresh(note=f"now drag the correct box for <b>#{self.selected}</b> on this frame" if self.selected
                     else "click an object in Select mode first")

    def _do(self, action, clear=False):
        """Run an edit, show progress, and never let an error kill the screen."""
        try:
            self.status.value = "⏳ working…"
            action()
            if clear: self.selected = None
            self.refresh()
        except Exception as e:
            self.refresh(note=f"<span style='color:#c00'>{type(e).__name__}: {e}</span>")

    def _on_event(self, change):
        ev = change["new"]
        if not ev: return
        t, s = self.slider.value, self._scale
        if ev["kind"] == "box":
            box = [ev["x0"] / s, ev["y0"] / s, ev["x1"] / s, ev["y1"] / s]
            if self._rebox and self.selected:
                self._rebox = False
                self._do(lambda: self.set_box(self.selected, t, box))
            else:
                self._do(lambda: self.add_box(t, box))
        else:
            x, y = min(int(ev["x"] / s), self.W - 1), min(int(ev["y"] / s), self.H - 1)
            if self.mode.value == "select":
                self.selected = int(self.ids[t, y, x]) or None
                self.refresh()
            elif not self.selected:
                self.refresh(note="click an object in Select mode first")
            else:
                self._do(lambda: self.add_click(self.selected, t, x, y, positive=self.mode.value == "pos"))

def label(clip=None, size=None, ckpt=None):
    """Open the labelling screen. Pick a clip from the menu, or pass a clip name."""
    clip = clip.name if isinstance(clip, Clip) else clip
    names = [c for c in clips_table()["clip"]]
    if not names:
        print("No clips yet: add videos and cut clips first (Steps 1 and 2)."); return
    def menu():
        t = clips_table().set_index("clip")
        mark = {"done": "✅ done ", "draft": "✏️ draft", "none": "⬜ to do"}
        return [(f"{mark[t.loc[n, 'labels']]} · {t.loc[n, 'split'] or '?'} · {n}", n) for n in names]
    pick = widgets.Dropdown(options=menu(), value=clip or names[0], description="clip", layout=widgets.Layout(width="620px"))
    box, current = widgets.VBox([pick, widgets.HTML("loading…")]), [None]
    def refresh_menu():
        pick.unobserve(open_clip, names="value")
        keep = pick.value; pick.options = menu(); pick.value = keep
        pick.observe(open_clip, names="value")
    def open_clip(change=None):
        if current[0] is not None: current[0].close()
        refresh_menu()
        current[0] = LabelEditor(pick.value, size, ckpt, on_status=refresh_menu)
        box.children = (pick, current[0].view)
    pick.observe(open_clip, names="value")
    display(box)
    open_clip()

def draft_labels(model, clips=None, overwrite=False):
    """Let an automatic model take the first pass. Drafts are not used for training or scoring until a person
    opens them in label(), fixes what is wrong and presses 'mark done'."""
    t = clips_table()             # never touches clips a person has already approved
    todo = clips or [Clip(n) for n, lab in zip(t["clip"], t["labels"]) if lab == "none" or (overwrite and lab == "draft")]
    model.prepare()
    for clip in tqdm(todo, desc="drafting"):
        back = getattr(model, "backward", None)
        if back is not None: model.backward = True          # also label frames before the first detection
        try:
            ids = np.asarray(model.predict(clip))
        finally:
            if back is not None: model.backward = back
        clip.save_labels(ids)
        _save_prompts(clip.name, getattr(model, "last_prompts", []))
        set_meta(clip.name, status="draft")
    print(f"drafted {len(todo)} clip(s). Next: label() to check each one.")

def mark_done(clips):
    """Approve clips from code (for labels you have already checked some other way)."""
    for c in clips: set_meta(c if isinstance(c, str) else c.name, status="done")

def ruler(clip, frame=0, step=100):
    """No-screen fallback: show a frame with a pixel grid so box corners can be read off for ClipLabels.add_box."""
    c = Clip(clip) if isinstance(clip, str) else clip
    lab = c.labels()
    img = overlay(c.image(frame), lab[frame]) if lab is not None else c.image(frame)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(16, 16 * h / w))
    ax.imshow(img); ax.set_xticks(range(0, w, step)); ax.set_yticks(range(0, h, step))
    ax.grid(color="white", linewidth=0.6, alpha=0.7); ax.tick_params(labelsize=7, labeltop=True, labelright=True)
    plt.show()

