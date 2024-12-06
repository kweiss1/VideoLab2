from collections import defaultdict

import torch

from apex.transformer import parallel_state


class GradScaler:
    """
    Gradient scaler for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.
    """

    def __init__(
        self, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True
    ):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._per_optimizer_states = defaultdict(dict)

    def _unscale_grads_(self, optimizer, *args):
        if getattr(optimizer, "_custom_amp_unscale_grads", False):
            return optimizer.unscale_grads(*args)
        else:
            # Perform unscaling manually here if required
            pass

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        found_inf = torch.tensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
        return retval

    def update(self, new_scale=None):
        """
        Updates the scale factor.
        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.
        Passing ``new_scale`` sets the new scale value manually.
        """
        if not self._enabled:
            return

        if new_scale is not None:
            # Accept a new user-defined scale.
            self._scale = new_scale
        else:
            # Handle gradient overflow detection manually if required.
            pass

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(dict)
