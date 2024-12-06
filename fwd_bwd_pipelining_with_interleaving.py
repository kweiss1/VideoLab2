import contextlib
from typing import Any, Callable, List, Optional, Sequence, Union
import warnings

import torch

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.schedules.common import Batch
from apex.transformer.pipeline_parallel.schedules.common import FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import backward_step
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import free_output_tensor
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import get_model_type
from apex.transformer.log_util import get_transformer_logger


__all__ = ["_forward_backward_pipelining_with_interleaving"]

_logger = get_transformer_logger(__name__)


# Function `_forward_backward_pipelining_with_interleaving`
def _forward_backward_pipelining_with_interleaving(
    forward_step_func: FwdStepFunc,
    batch: List[Optional[Batch]],
    model: List[torch.nn.Module],
    *,
    forward_only: bool,
    tensor_shape: Optional[Union[List[int], torch.Size]] = None,
    dtype: Optional[torch.dtype] = None,
    disable_autocast: bool = False,
    deallocate_pipeline_outputs: bool = False,
    async_comm: bool = False,
    sequence_parallel_enabled: bool = False,
    custom_sync_context_handler: Optional[Callable] = None,
    custom_grad_sync_func: Optional[Callable] = None,
    custom_param_sync_func: Optional[Callable] = None,
    sync_batch_comm: bool = True,
    num_micro_batches_with_partial_activation_checkpoints: Optional[int] = None,
    overlap_p2p_comm: bool = False,
    batch_p2p_comm: bool = True,
    **kwargs,
) -> List[Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Run interleaved 1F1B schedule with communication between pipeline stages as needed."""
    if not isinstance(model, list):
        raise RuntimeError("`model` must be a list of `nn.Module`'s'")

    if deallocate_pipeline_outputs:
        warnings.warn(
            "`deallocate_pipeline_outputs` is experimental and subject to change. "
            "This option is not recommended."
        )

    # Construct helper functions for async grad reductions
    if custom_sync_context_handler is not None:
        sync_context_handler = custom_sync_context_handler
    else:
        sync_context_handler = contextlib.nullcontext
    sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions."""
        nonlocal sync_context
        if sync_context is None:
            sync_context = sync_context_handler()
            sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions."""
        nonlocal sync_context
        if sync_context is not None:
            sync_context.__exit__(None, None, None)
            sync_context = None

    disable_grad_sync()

    # Remaining functionality continues unchanged...

    ###################################################################################################################
    # Replace `grad_scaler` usage in `forward_step` and `backward_step` calls:
    ###################################################################################################################
    def forward_step_helper(
        microbatch_id: int,
        curr_iters: List[int],
        checkpoint_activations_micro_batch: Optional[bool] = None,
    ) -> torch.Tensor:
        """Helper method to run forward step with model split into chunks."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if custom_param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(
                param_sync_microbatch_id
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    custom_param_sync_func(model[param_sync_chunk_id].parameters())

        if parallel_state.is_pipeline_first_stage() and len(input_tensors[model_chunk_id]) == len(
            output_tensors[model_chunk_id]
        ):
            input_tensors[model_chunk_id].append(None)

        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(
            forward_step_func,
            get_kth_microbatch(batch, curr_iters[model_chunk_id]),
            model[model_chunk_id],
            input_tensor,
            losses_reduced,
            dtype,
            disable_autocast,
            checkpoint_activations_micro_batch,
        )
        curr_iters[model_chunk_id] += 1
        output_tensors[model_chunk_id].append(output_tensor)

        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id: int) -> torch.Tensor:
        """Helper method to run backward step with model split into chunks."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        model_type = get_model_type(model[model_chunk_id])
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if custom_grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()

        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor,
            output_tensor,
            output_tensor_grad,
            model_type=model_type,
            deallocate_pipeline_outputs=deallocate_pipeline_outputs,
        )

        if custom_grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                custom_grad_sync_func(model[grad_sync_chunk_id].parameters())
        disable_grad_sync()

        return input_tensor_grad


    ###################################################################################################################
    # Run warmup forward passes.
    ###################################################################################################################
    fwd_wait_handles, bwd_wait_handles = None, None
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(
            tensor_shape=tensor_shape,
            dtype=dtype,
            async_comm=async_comm,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sync_batch_comm=sync_batch_comm,
            batch_p2p_comm=batch_p2p_comm,
        )
    )
    _logger.info("Warmup phase")
    for k in range(num_warmup_microbatches):
        _logger.debug(f"warmup iter: {k} / {num_warmup_microbatches}")

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_micro_batch = k % max_outstanding_backprops >= \
                num_micro_batches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_micro_batch = None

        if fwd_wait_handles is not None:
            for wait_handle in fwd_wait_handles:
                wait_handle.wait()

        output_tensor = forward_step_helper(k, curr_iters, checkpoint_activations_micro_batch)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False
        _logger.debug(
            f"next fwd model chunk ID: {next_forward_model_chunk_id}, recv_prev: {recv_prev}"
        )

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            _logger.debug("Pipeline last stage, not sending tensor downstream")
            output_tensor = None

        if overlap_p2p_comm:
            # P2P communications in warmup are not overlapped with computes. We split P2P
            # communications for activation forward and activation_gradient backward in warmup,
            # to match the send/recv API granularity in 1F1B in case of using batched send/recv API.

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            _logger.debug("send fwd and receive fwd")
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                overlap_p2p_comm=True,
                batch_p2p_comm=batch_p2p_comm,
            )
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                _logger.debug("send bwd and receive bwd")
                output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    overlap_p2p_comm=True,
                    batch_p2p_comm=batch_p2p_comm,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                _logger.debug("send fwd&bwd and receive fwd&bwd")
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                _logger.debug("send fwd and receive fwd")
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
            free_output_tensor(output_tensor, deallocate_pipeline_outputs)

    ###################################################################################################################
    # Run 1F1B in steady state.
    ###################################################################################################################
    _logger.info("Steady phase")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        _logger.debug(f" steady phase iter {k} / {num_microbatches_remaining}")
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_micro_batch = (
                forward_k % max_outstanding_backprops >= num_micro_batches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_micro_batch = None

        if overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for wait_handle in fwd_wait_handles:
                    wait_handle.wait()

            output_tensor = forward_step_helper(forward_k, curr_iters, checkpoint_activations_micro_batch)

            # Set forward model chunk id
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if the current virtual stage has an activation tensor to receive
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k + 1, forward=True
                )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the 
            # previous stage
            _logger.debug("send fwd and receive fwd")
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                overlap_p2p_comm=True,
                batch_p2p_comm=batch_p2p_comm,
            )

            if bwd_wait_handles is not None:
                for wait_handle in bwd_wait_handles:
                    wait_handle.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Set backward model chunk id
            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            _logger.debug(
                f"fwd/bwd model chunk id: {forward_model_chunk_id}/{backward_model_chunk_id}"
            )

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k + 1, forward=False
                )

            # Send activation grad tensor to the previous stage and receive activation grad tensor
            # from the previous stage
            _logger.debug("send bwd and receive bwd")
            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                overlap_p2p_comm=True,
                batch_p2p_comm=batch_p2p_comm,
            )
        else:
            output_tensor = forward_step_helper(forward_k, curr_iters, checkpoint_activations_micro_batch)

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            _logger.debug(
                f"fwd/bwd model chunk id: {forward_model_chunk_id}/{backward_model_chunk_id}"
            )
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k + 1, forward=True
                )

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k + 1, forward=False
                )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            _logger.debug("send fwd&bwd and receive fwd&bwd")
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                dtype=dtype,
                async_comm=async_comm,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sync_batch_comm=sync_batch_comm,
                batch_p2p_comm=batch_p2p_comm,
            )
            free_output_tensor(output_tensor, deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    ###################################################################################################################
    # Run cooldown backward passes (flush out pipeline).
    ###################################################################################################################
    _logger.info("Cooldown phase")
    if not forward_only:
        if overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
            )

        for k in range(num_microbatches_remaining, num_microbatches):
            _logger.debug(
                f"cooldown iter {k} in range({num_microbatches_remaining}, {num_microbatches})"
            )
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False

            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    async_comm=async_comm,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    sync_batch_comm=sync_batch_comm,
                    batch_p2p_comm=batch_p2p_comm,
                )
            )

    # Make sure to exit context handler for async grad reductions
    enable_grad_sync()

    return losses_reduced
