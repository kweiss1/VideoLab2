from typing import Any, Callable, Dict, List, Tuple, Union, Optional, Sequence

import torch
from torch.autograd.variable import Variable

from apex.normalization.fused_layer_norm import FusedLayerNorm
from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel.p2p_communication import FutureTensor
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import unwrap_model
from apex.transformer.pipeline_parallel.utils import get_model_type
from apex.transformer.tensor_parallel.layers import (
    set_defaults_if_not_set_tensor_model_parallel_attributes,
)
from apex.transformer.log_util import get_transformer_logger


_logger = get_transformer_logger(__name__)


Batch = Union[torch.Tensor, FutureTensor, List[Union[torch.Tensor, FutureTensor]], Tuple[Union[torch.Tensor, FutureTensor], ...]]
LossFunc = Callable[[torch.Tensor], torch.Tensor]
FwdStepFunc = Callable[
    [Optional[Batch], torch.nn.Module], Tuple[torch.Tensor, LossFunc]
]


def build_model(
    model_provider_func: Callable[[Any, Dict[str, Any]], torch.nn.Module],
    wrap_with_ddp: bool = True,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    model_type: ModelType = ModelType.encoder_or_decoder,
    *args: Any,
    **kwargs: Any,
) -> List[torch.nn.Module]:
    """Build the model satisfying pipeline model parallel requirements."""
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and virtual_pipeline_model_parallel_size is not None
    ):
        model = []
        for i in range(virtual_pipeline_model_parallel_size):
            cur_args = args
            cur_kwargs = kwargs
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            cur_kwargs.update(
                {"pre_process": pre_process, "post_process": post_process}
            )
            this_model = model_provider_func(*cur_args, **cur_kwargs)
            model.append(this_model)
    else:
        cur_args = args
        cur_kwargs = kwargs
        if model_type == ModelType.encoder_or_decoder:
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            cur_kwargs.update(
                {"pre_process": pre_process, "post_process": post_process}
            )
            model = model_provider_func(*cur_args, **cur_kwargs)
        elif model_type == ModelType.encoder_and_decoder:
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            add_encoder, add_decoder = True, True
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                split_rank = parallel_state.get_pipeline_model_parallel_split_rank()
                if split_rank is None:
                    raise RuntimeError(
                        "Split rank needs to be specified for model with both encoder and decoder."
                    )
                rank = parallel_state.get_pipeline_model_parallel_rank()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = rank == (split_rank - 1) or rank == (world_size - 1)
                add_encoder = parallel_state.is_pipeline_stage_before_split()
                add_decoder = parallel_state.is_pipeline_stage_after_split()
            cur_kwargs.update(
                {
                    "pre_process": pre_process,
                    "post_process": post_process,
                    "add_encoder": add_encoder,
                    "add_decoder": add_decoder,
                }
            )
            model = model_provider_func(*cur_args, **cur_kwargs)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    for model_module in model:
        for param in model_module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    if (
        parallel_state.model_parallel_is_initialized()
        and parallel_state.get_data_parallel_rank() == 0
    ):
        msg = " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            _calc_number_of_params(model),
        )
        print(msg, flush=True)

    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    if wrap_with_ddp:
        i = torch.cuda.current_device()
        model = [
            torch.nn.parallel.distributed.DistributedDataParallel(
                model_module,
                device_ids=[i],
                output_device=i,
                process_group=parallel_state.get_data_parallel_group(),
            )
            for model_module in model
        ]
    return model


def _calc_number_of_params(model: List[torch.nn.Module]) -> int:
    return sum(
        [
            sum([p.nelement() for p in model_module.parameters()])
            for model_module in model
        ]
    )


def _get_params_for_weight_decay_optimization(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    *,
    no_weight_decay_modules=(FusedLayerNorm,),
) -> Dict[str, torch.nn.Parameter]:
    modules = listify_model(model)
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, no_weight_decay_modules):
                no_weight_decay_params["params"].extend(
                    [p for p in list(module_._parameters.values()) if p is not None]
                )
            else:
                weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n != "bias"
                    ]
                )
                no_weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n == "bias"
                    ]
                )

    return weight_decay_params, no_weight_decay_params


def free_output_tensor(
    output_tensors: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    deallocate_pipeline_outputs: bool = False,
) -> None:
    if not deallocate_pipeline_outputs:
        return
    if output_tensors is None:
        return
    if isinstance(output_tensors, torch.Tensor):
        output_tensors = [output_tensors]
    for output_tensor in output_tensors:
        output_tensor.data = torch.cuda.FloatTensor([0])


def custom_backward(output: torch.Tensor, grad_output: Optional[torch.Tensor]) -> None:
    if grad_output is None:
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    forward_step_func: FwdStepFunc,
    batch: Optional[Batch],
    model: torch.nn.Module,
    input_tensor: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    losses_reduced: List[torch.Tensor],
    dtype: torch.dtype,
    disable_autocast: bool = False,
    checkpoint_activations_micro_batch: Optional[bool] = None,
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    unwrapped_model = unwrap_model(model)
    model_type = get_model_type(unwrapped_model)
    unwrap_output_tensor = not isinstance(input_tensor, list)
    if unwrap_output_tensor:
        input_tensor = [input_tensor]

    input_tensor = [inp.get() if isinstance(inp, FutureTensor) else inp for inp in input_tensor]

    unwrapped_model.set_input_tensor(input_tensor)
    with torch.no_grad():
        if checkpoint_activations_micro_batch is None:
            output_tensor, loss_func = forward_step_func(batch, model)
        else:
            output_tensor, loss_func = forward_step_func(batch, model, checkpoint_activations_micro_batch)
        if parallel_state.is_pipeline_last_stage():
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / get_num_microbatches()
            losses_reduced.append(loss_reduced)

    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(
    input_tensor: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
    output_tensor_grad: Optional[torch.Tensor],
    model_type: ModelType,
    *,
    deallocate_pipeline_outputs: bool = False,
) -> Union[None, torch.Tensor, Sequence[torch.Tensor]]:
    """Backward step through passed-in output tensor."""
    unwrap_input_tensor_grad = not isinstance(input_tensor, list)
    if unwrap_input_tensor_grad:
        input_tensor = [input_tensor]

    input_tensor = [inp.get() if isinstance(inp, FutureTensor) else inp for inp in input_tensor]

    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]

    output_tensor = [out.get() if isinstance(out, FutureTensor) else out for out in output_tensor]

    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    output_tensor_grad = [ogr.get() if isinstance(ogr, FutureTensor) else ogr for ogr in output_tensor_grad]

    if deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            input_tensor_grad.append(None if x is None else x.grad)

    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])

    return input_tensor_grad[0] if unwrap_input_tensor_grad else input_tensor_grad
