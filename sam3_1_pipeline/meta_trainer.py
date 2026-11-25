"""Start Meta's trainer with two corrections, so that fine-tuning can run.

finetune.py calls this file with the arguments of `python -m sam3.train.train`:

  python meta_trainer.py -c <config> --use-cluster 0 --num-gpus 1

1. The vision backbone can be trained. Since Meta's SAM 3.1 release (27 March 2026), the
MLP inside the vision backbone (sam3/model/vitdet.py) calls a fused operation that only
works for inference. Meta's own trainer therefore stops on the first training step with
"ValueError: Expected grad to be disabled." This file gives that MLP its ordinary
forward pass back (linear layer, activation, linear layer: the same computation, and
what the code did before that release).

2. A training step whose numbers are not finite is left out. Now and then a batch
gives NaN or infinite values, in the model's output or in the loss. Meta's trainer then
stops, with "ValueError: matrix contains invalid numeric entries" from the step that
pairs predictions with labels, or with "FloatingPointError: Loss is nan", and the epoch
it was in is lost. Here such a step passes no gradient to the weights, is listed in the
run's logs/skipped_steps.txt, and training goes on. Training still stops when more
than MAX_SKIPPED_STEPS steps were left out, or when the weights themselves are no
longer finite.

Otherwise Meta's trainer runs unchanged. Nothing in the installed sam3 package is
edited, and a sam3 version whose MLP does not call the fused operation keeps its MLP.
"""

from __future__ import annotations

import inspect
import logging
import runpy
from pathlib import Path

MAX_SKIPPED_STEPS = 20
# scipy's words when the costs of pairing predictions with labels hold NaN or -inf
NOT_FINITE = "invalid numeric entries"
SKIPPED: list[str] = []  # one line per step left out by this process


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


def leave_step_out(trainer, error: Exception, epoch: int, step: int) -> None:
    """Called when a training step failed on numbers that are not finite: check that
    training can go on, write the step down, and hand Meta's training loop a gradient of
    zero to finish the step with.
    """
    import torch

    reason = f"{type(error).__name__}: {error}"
    where = f"epoch {epoch}, step {step} of the epoch"
    if not all(bool(torch.isfinite(p).all()) for p in trainer.model.parameters()):
        raise RuntimeError(
            f"meta_trainer: {where}: the model's weights are no longer finite, so "
            f"training stops ({reason})"
        ) from error
    if len(SKIPPED) >= MAX_SKIPPED_STEPS:
        raise RuntimeError(
            f"meta_trainer: {where}: more than {MAX_SKIPPED_STEPS} training steps had "
            f"numbers that are not finite, so training stops ({reason})"
        ) from error
    # No gradient of this step may reach the weights. Meta's loop goes on to its
    # optimizer step, and the gradient scaler there needs one gradient to look at: the
    # smallest tensor the optimizer trains gets a gradient of zero (the optimizer moves
    # it as on any step, by the learning rate at most), every other weight has none and
    # stays exactly as it is.
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
    SKIPPED.append(f"{epoch}\t{step}\t{reason}")
    try:
        log = Path(trainer.logging_conf.log_dir) / "skipped_steps.txt"
        with log.open("a") as f:
            f.write(SKIPPED[-1] + "\n")
    except (AttributeError, OSError, TypeError):
        pass  # the line below still reaches the job's log
    logging.warning(
        "meta_trainer: %s left out, its numbers were not finite (%s). Left out so "
        "far: %d.",
        where,
        reason,
        len(SKIPPED),
    )


def skip_steps_that_are_not_finite() -> bool:
    """Wrap the training step of Meta's trainer so that a step whose numbers are not
    finite is left out (leave_step_out) in place of ending the training. True if the
    trainer's step was found.
    """
    from sam3.train import trainer

    cls = getattr(trainer, "Trainer", None)
    run_step = getattr(cls, "_run_step", None)
    if run_step is None:
        return False

    def run_step_or_skip(self, batch, *args, **kwargs):
        # count the steps of the epoch as Meta's log does: "Train Epoch: [15][ 80/256]"
        epoch = int(self.epoch)
        last_epoch, last_step = getattr(self, "_step_in_epoch", (None, -1))
        step = last_step + 1 if epoch == last_epoch else 0
        self._step_in_epoch = (epoch, step)
        try:
            return run_step(self, batch, *args, **kwargs)
        except (FloatingPointError, ValueError) as error:
            if isinstance(error, ValueError) and NOT_FINITE not in str(error):
                raise
            leave_step_out(self, error, epoch, step)

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
        print("meta_trainer: this sam3 version needs no correction.", flush=True)
    report_matching_inputs()
    if skip_steps_that_are_not_finite():
        print(
            "meta_trainer: a training step whose numbers are not finite is left out "
            f"(at most {MAX_SKIPPED_STEPS}).",
            flush=True,
        )
    try:
        runpy.run_module("sam3.train.train", run_name="__main__", alter_sys=True)
    finally:
        if SKIPPED:
            print(
                f"meta_trainer: {len(SKIPPED)} training step(s) left out, their "
                "numbers were not finite.",
                flush=True,
            )
