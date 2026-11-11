"""Start Meta's trainer with one correction, so that fine-tuning can run. finetune.py calls this file.

    python meta_trainer.py -c <config> --use-cluster 0 --num-gpus 1        the arguments of python -m sam3.train.train

Since Meta's SAM 3.1 release (27 March 2026), the MLP inside the vision backbone (sam3/model/vitdet.py) calls a
fused operation that only works for inference. Meta's own trainer therefore stops on the first training step with
"ValueError: Expected grad to be disabled." This file gives that MLP its ordinary forward pass back (linear layer,
activation, linear layer: the same computation, and what the code did before that release), then runs Meta's
trainer unchanged. Nothing in the installed sam3 package is edited, and a sam3 version whose MLP does not call the
fused operation is left as it is.
"""
from __future__ import annotations

import inspect
import runpy


def trainable_mlp_forward(self, x):
    """The MLP of sam3.model.vitdet with ordinary layers: fc1, activation, dropout, norm, fc2, dropout."""
    x = self.act(self.fc1(x))
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


def correct_vision_mlp() -> bool:
    """Swap the inference-only forward of sam3's vision-backbone MLP for the trainable one. True if it was swapped."""
    from sam3.model import vitdet
    mlp = getattr(vitdet, 'Mlp', None)
    if mlp is None:
        return False
    try:
        source = inspect.getsource(mlp.forward)
    except (OSError, TypeError):
        return False
    if 'addmm_act' not in source:
        return False
    mlp.forward = trainable_mlp_forward
    return True


if __name__ == '__main__':
    if correct_vision_mlp():
        print("meta_trainer: the vision backbone's MLP runs with ordinary layers, so it can be trained "
              '(sam3 ships an inference-only version).', flush=True)
    else:
        print('meta_trainer: this sam3 version needs no correction.', flush=True)
    runpy.run_module('sam3.train.train', run_name='__main__', alter_sys=True)
