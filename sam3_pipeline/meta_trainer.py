"""Start Meta's trainer with four corrections, so that fine-tuning can run.

finetune.py calls this file with the arguments of `python -m sam3.train.train`:

  python meta_trainer.py -c <config> --use-cluster 0 --num-gpus 1

1. The vision backbone can be trained. Since Meta's SAM 3.1 release (27 March 2026), the
MLP inside the vision backbone (sam3/model/vitdet.py) calls a fused operation that only
works for inference. Meta's own trainer therefore stops on the first training step with
"ValueError: Expected grad to be disabled." This file gives that MLP its ordinary
forward pass back (linear layer, activation, linear layer: the same computation, and
what the code did before that release).

2. The presence loss has a slope everywhere. The loss for "is the thing in this image at
all" is a focal loss with gamma = 0, computed by Meta's GPU kernel
(sam3/train/loss/sigmoid_focal_loss.py). For the slope, the kernel computes
(1 - p) ** (gamma - 1), where p says how right the prediction is. With gamma = 0 that
is 1 / (1 - p): once a prediction is right to the last digit of float32 (a presence
logit above about 16.6, or below about -17.3 for an image without the thing), it
divides by zero, and the gradient of the whole step is NaN although the loss is an
ordinary number. With gamma = 0 a focal loss is the cross entropy weighted by alpha.
Here it is computed as that: the same value up to rounding, and a slope that is
finite everywhere. Any other gamma goes to Meta's kernel as before.

3. The gradient scale stays usable. Meta's trainer multiplies the loss by a scale before
the gradients are computed and divides the gradients by it afterwards (PyTorch's
GradScaler). After a step whose gradient is not finite, PyTorch skips the step's update
and halves the scale, and it doubles the scale again only after 2,000 steps in a row
without such a step. When such steps keep coming, the scale only shrinks, and the
smaller it is, the less PyTorch's check is worth, because it looks at the gradients
while they are still multiplied by the scale: a gradient too large for float32 passes
as an ordinary number and overflows only when it is divided by the scale (after 144
halvings 1 / scale is itself infinite, and every gradient does). The optimizer then
writes NaN into the weights, and the step after that stops the trainer with
"ValueError: matrix contains invalid numeric entries". Small gradients, for their part,
are rounded to zero. Here the scale never falls below MIN_GRADIENT_SCALE. Training runs
in bfloat16, which holds numbers as small as float32 does, so a scale below 1 helps
nothing. And whatever the scale, a gradient that is not finite never reaches the
optimizer.

4. A training step whose numbers are not finite changes no weights and is written down.
Two things can be not finite. The model's output or the loss: Meta's trainer then stops,
with "ValueError: matrix contains invalid numeric entries" from the step that pairs
predictions with labels, or with "FloatingPointError: Loss is nan", and the epoch it was
in is lost. Here that step is left out. Or the gradient: PyTorch then leaves the weights
as they are, without a word. Either way the step is listed in the run's
logs/skipped_steps.txt, and training goes on. Training still stops when the output or
the loss was not finite on more than MAX_SKIPPED_STEPS steps, or when the weights
themselves are no longer finite.

Otherwise Meta's trainer runs unchanged. Nothing in the installed sam3 package is
edited, and a sam3 version whose MLP does not call the fused operation keeps its MLP.
"""

from __future__ import annotations

import inspect
import logging
import runpy
from pathlib import Path

MAX_SKIPPED_STEPS = 20
MIN_GRADIENT_SCALE = 1.0
# scipy's words when the costs of pairing predictions with labels hold NaN or -inf
NOT_FINITE = "invalid numeric entries"
# one line per step of this process that changed no weights
SKIPPED: list[str] = []  # the output or the loss was not finite
NO_UPDATE: list[str] = []  # the gradient was not finite


class Step:
    """The training step a trainer is on."""

    def __init__(self):
        self.epoch, self.number = -1, -1
        self.left_out = False  # its output or loss was not finite
        self.listed = False  # it is in skipped_steps.txt
        # weight tensors that were not finite before the trainer's first step
        self.not_finite_at_start: frozenset[str] = frozenset()

    def __str__(self) -> str:
        return f"epoch {self.epoch}, step {self.number} of the epoch"


def trainable_mlp_forward(self, x):
    """The MLP of sam3.model.vitdet with ordinary layers: fc1, activation, dropout,
    norm, fc2, dropout.
    """
    x = self.act(self.fc1(x))
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


def correct_vision_mlp() -> bool:
    """Swap the inference-only forward of sam3's vision-backbone MLP for the trainable
    one. True if it was swapped.
    """
    from sam3.model import vitdet

    mlp = getattr(vitdet, "Mlp", None)
    if mlp is None:
        return False
    try:
        source = inspect.getsource(mlp.forward)
    except (OSError, TypeError):
        return False
    if "addmm_act" not in source:
        return False
    mlp.forward = trainable_mlp_forward
    return True


def correct_focal_loss_with_gamma_zero() -> bool:
    """With gamma = 0, compute Meta's focal loss as the cross entropy weighted by alpha,
    which is what it then is, in place of Meta's GPU kernel, whose slope is NaN for a
    prediction that is exactly right. Any other gamma goes to the kernel. True if the
    kernel's two entry points were found.
    """
    try:
        from sam3.train.loss import loss_fns
    except ImportError:
        return False
    import torch.nn.functional as F

    each_kernel = getattr(loss_fns, "triton_sigmoid_focal_loss", None)
    summed_kernel = getattr(loss_fns, "triton_sigmoid_focal_loss_reduce", None)
    if each_kernel is None or summed_kernel is None:
        return False

    def cross_entropy(inputs, targets, alpha):
        # float32, like the kernel, whatever precision the model runs in
        inputs, targets = inputs.float(), targets.float()
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        return (alpha * targets + (1 - alpha) * (1 - targets)) * loss

    def each(inputs, targets, alpha=0.25, gamma=2):
        if gamma == 0:
            return cross_entropy(inputs, targets, alpha)
        return each_kernel(inputs, targets, alpha, gamma)

    def summed(inputs, targets, alpha=0.25, gamma=2):
        if gamma == 0:
            return cross_entropy(inputs, targets, alpha).sum()
        return summed_kernel(inputs, targets, alpha, gamma)

    loss_fns.triton_sigmoid_focal_loss = each
    loss_fns.triton_sigmoid_focal_loss_reduce = summed
    return True


def describe_matching_inputs(outputs, batched_targets) -> str:
    """What went into the pairing of predictions with labels, for the error text: how
    many predicted scores, predicted boxes and labelled boxes are not finite, and how
    many boxes have no area (a predicted box and a labelled box without area cannot be
    compared either).
    """
    import torch
    from sam3.model.box_ops import box_area, box_cxcywh_to_xyxy

    scores, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    padded = batched_targets["boxes_padded"]
    labelled = torch.cat(
        [padded[i, :n] for i, n in enumerate(batched_targets["num_boxes"].tolist())]
    )

    def not_finite(tensor) -> str:
        return f"{int((~torch.isfinite(tensor)).sum())} of {tensor.numel()} not finite"

    def no_area(cxcywh) -> str:
        flat = box_area(box_cxcywh_to_xyxy(cxcywh.float().reshape(-1, 4))) == 0
        return f"{int(flat.sum())} of {flat.numel()} with no area"

    return (
        f"predicted scores: {not_finite(scores)}; "
        f"predicted boxes: {not_finite(boxes)}, {no_area(boxes)}; "
        f"labelled boxes: {not_finite(labelled)}, {no_area(labelled)}"
    )


def report_matching_inputs() -> bool:
    """When pairing predictions with labels fails on numbers that are not finite, add
    to the error what went in (describe_matching_inputs). True if Meta's matcher was
    found.
    """
    from sam3.train import matcher

    cls = getattr(matcher, "BinaryHungarianMatcherV2", None)
    if cls is None:
        return False
    forward = cls.forward

    def forward_with_report(self, outputs, batched_targets, *args, **kwargs):
        try:
            return forward(self, outputs, batched_targets, *args, **kwargs)
        except ValueError as error:
            if NOT_FINITE not in str(error):
                raise
            try:
                inputs = describe_matching_inputs(outputs, batched_targets)
            except Exception:  # the report must not hide the error itself
                inputs = ""
            if not inputs:
                raise
            raise ValueError(f"{error} ({inputs})") from error

    cls.forward = forward_with_report
    return True


def write_down(trainer, step: Step, reason: str) -> str:
    """List the step in the run's logs/skipped_steps.txt: epoch, step of the epoch,
    reason. Returns the line.
    """
    line = f"{step.epoch}\t{step.number}\t{reason}"
    step.listed = True
    try:
        log = Path(trainer.logging_conf.log_dir) / "skipped_steps.txt"
        with log.open("a") as f:
            f.write(line + "\n")
    except (AttributeError, OSError, TypeError):
        pass  # the warning that follows still reaches the job's log
    return line


def weights_not_finite(model) -> list[str]:
    """The names of the model's weight tensors that hold NaN or an infinite value."""
    import torch

    return [
        name
        for name, weight in model.named_parameters()
        if not bool(torch.isfinite(weight).all())
    ]


def gradients_are_finite(optimizer) -> bool:
    """True if none of the gradients the optimizer would use holds NaN or an infinite
    value. The sum of a tensor is not finite as soon as one of its values is not.
    """
    import torch

    sums = [
        weight.grad.sum()
        for group in optimizer.param_groups
        for weight in group["params"]
        if weight.grad is not None
    ]
    if not sums:
        return True
    return bool(torch.isfinite(torch.stack([s.to(sums[0].device) for s in sums])).all())


def keep_gradient_scale(scaler) -> float | None:
    """Raise the scale of the trainer's gradient scaler to MIN_GRADIENT_SCALE if it is
    below. Returns the scale it had if it was raised, else None. A scaler that is
    switched off reports a scale of 1 and is left as it is.
    """
    scale = scaler.get_scale()
    if scale >= MIN_GRADIENT_SCALE:
        return None
    scaler.load_state_dict({**scaler.state_dict(), "scale": MIN_GRADIENT_SCALE})
    return scale


def gradient_not_finite(trainer, step: Step, where: str) -> None:
    """Write down a step whose gradient is not finite. Its weights stay as they are:
    PyTorch's gradient scaler skips the update of such a step, and the optimizer's own
    check (keep_gradients_finite) drops what the scaler lets through.
    """
    if step.listed:
        return
    NO_UPDATE.append(write_down(trainer, step, f"gradient not finite {where}"))
    logging.warning(
        "meta_trainer: %s changes no weights, its gradient is not finite (%s). Such "
        "steps so far: %d.",
        step,
        where,
        len(NO_UPDATE),
    )


def keep_gradients_finite(trainer, step: Step) -> None:
    """Give the trainer's optimizer a check that runs each time it is about to change
    the weights: if a gradient is not finite, or the step was left out, every gradient
    is taken away, and the optimizer changes nothing.
    """

    def check(optimizer, args, kwargs) -> None:
        if not step.left_out and gradients_are_finite(optimizer):
            return
        for group in optimizer.param_groups:
            for weight in group["params"]:
                weight.grad = None
        if not step.left_out:
            gradient_not_finite(trainer, step, "at the optimizer")

    trainer.optim.optimizer.register_step_pre_hook(check)


def leave_step_out(trainer, step: Step, error: Exception) -> None:
    """Called when a training step failed on an output or a loss that is not finite:
    check that training can go on, write the step down, and hand Meta's training loop a
    gradient of zero to finish the step with.
    """
    import torch

    reason = f"{type(error).__name__}: {error}"
    bad = [
        name
        for name in weights_not_finite(trainer.model)
        if name not in step.not_finite_at_start
    ]
    if bad:
        raise RuntimeError(
            f"meta_trainer: {step}: the model's weights are no longer finite "
            f"(tensors with NaN or infinity: {len(bad)}, the first is {bad[0]}; "
            f"gradient scale {trainer.scaler.get_scale():g}), so training stops "
            f"({reason})"
        ) from error
    if len(SKIPPED) >= MAX_SKIPPED_STEPS:
        raise RuntimeError(
            f"meta_trainer: {step}: on more than {MAX_SKIPPED_STEPS} training steps "
            f"the output or the loss was not finite, so training stops ({reason})"
        ) from error
    # No gradient of this step may reach the weights. Meta's loop goes on to its
    # optimizer step, and the gradient scaler there needs one gradient to look at: the
    # smallest tensor the optimizer trains gets a gradient of zero. The optimizer's
    # check (keep_gradients_finite) takes it away again, so no weight moves.
    trainer.optim.zero_grad(set_to_none=True)
    trained = [
        p
        for group in trainer.optim.optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    ]
    smallest = min(trained, key=lambda p: p.numel())
    smallest.grad = torch.zeros_like(smallest)
    # on the first step of a process the scaler has not been used yet: this readies it
    trainer.scaler.scale(smallest.grad)
    step.left_out = True
    SKIPPED.append(write_down(trainer, step, reason))
    logging.warning(
        "meta_trainer: %s left out, its output or loss was not finite (%s). Left out "
        "so far: %d.",
        step,
        reason,
        len(SKIPPED),
    )


def skip_steps_that_are_not_finite() -> bool:
    """Wrap the training step of Meta's trainer: before it, the gradient scale is kept
    usable (keep_gradient_scale); a step whose output or loss is not finite is left out
    (leave_step_out) in place of ending the training; a step whose gradient is not
    finite is written down (gradient_not_finite). True if the trainer's step was found.
    """
    from sam3.train import trainer

    cls = getattr(trainer, "Trainer", None)
    run_step = getattr(cls, "_run_step", None)
    if run_step is None:
        return False

    def run_step_or_skip(self, batch, *args, **kwargs):
        step = getattr(self, "_meta_trainer_step", None)
        if step is None:
            step = self._meta_trainer_step = Step()
            keep_gradients_finite(self, step)
            step.not_finite_at_start = frozenset(weights_not_finite(self.model))
            if step.not_finite_at_start:
                print(
                    f"meta_trainer: {len(step.not_finite_at_start)} weight tensor(s) "
                    "hold NaN or infinity before the first training step, the first "
                    f"is {min(step.not_finite_at_start)}. If every step is left out, "
                    "the checkpoint the training started from is damaged.",
                    flush=True,
                )
        # count the steps of the epoch as Meta's log does: "Train Epoch: [15][ 80/256]"
        epoch = int(self.epoch)
        step.number = step.number + 1 if epoch == step.epoch else 0
        step.epoch, step.left_out, step.listed = epoch, False, False
        scale = keep_gradient_scale(self.scaler)
        # a scale of half the minimum is the ordinary case, one halving after a step
        # whose gradient was not finite; anything else was read from a checkpoint
        if scale is not None and not scale >= MIN_GRADIENT_SCALE / 2:
            print(
                f"meta_trainer: {step}: the gradient scale was {scale:g}; it is now "
                f"{MIN_GRADIENT_SCALE:g}.",
                flush=True,
            )
        try:
            result = run_step(self, batch, *args, **kwargs)
        except (FloatingPointError, ValueError) as error:
            if isinstance(error, ValueError) and NOT_FINITE not in str(error):
                raise
            leave_step_out(self, step, error)
            return None
        if not gradients_are_finite(self.optim.optimizer):
            gradient_not_finite(self, step, "after the backward pass")
        return result

    cls._run_step = run_step_or_skip
    return True


if __name__ == "__main__":
    if correct_vision_mlp():
        print(
            "meta_trainer: the vision backbone's MLP runs with ordinary layers, so it "
            "can be trained "
            "(sam3 ships an inference-only version).",
            flush=True,
        )
    else:
        print(
            "meta_trainer: the vision backbone's MLP of this sam3 version needs no "
            "correction.",
            flush=True,
        )
    if correct_focal_loss_with_gamma_zero():
        print(
            "meta_trainer: a focal loss with gamma = 0 (the presence loss) is computed "
            "as the weighted cross entropy it is, so its gradient is finite.",
            flush=True,
        )
    report_matching_inputs()
    if skip_steps_that_are_not_finite():
        print(
            "meta_trainer: the gradient scale stays at "
            f"{MIN_GRADIENT_SCALE:g} or above, and a training step whose numbers are "
            "not finite changes no weights (training stops after "
            f"{MAX_SKIPPED_STEPS} steps whose output or loss is not finite).",
            flush=True,
        )
    try:
        runpy.run_module("sam3.train.train", run_name="__main__", alter_sys=True)
    finally:
        if SKIPPED or NO_UPDATE:
            print(
                f"meta_trainer: {len(SKIPPED) + len(NO_UPDATE)} training step(s) "
                f"changed no weights: on {len(SKIPPED)} the output or the loss was "
                f"not finite, on {len(NO_UPDATE)} the gradient.",
                flush=True,
            )
