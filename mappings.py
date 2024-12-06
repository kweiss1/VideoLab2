# coding=utf-8
# Copyright (c) 2021-22, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from apex.transformer.parallel_state import get_tensor_model_parallel_group
from apex.transformer.parallel_state import get_tensor_model_parallel_world_size
from apex.transformer.parallel_state import get_tensor_model_parallel_rank
from apex.transformer.tensor_parallel.utils import split_tensor_along_last_dim

# Backward compatibility for older PyTorch versions
if "all_gather_into_tensor" not in dir(torch.distributed):
    def all_gather_into_tensor(output_tensor, input_tensor, group=None):
        tensors = [torch.empty_like(input_tensor) for _ in range(torch.distributed.get_world_size(group))]
        torch.distributed.all_gather(tensors, input_tensor, group=group)
        output_tensor.copy_(torch.cat(tensors, dim=0))
    torch.distributed.all_gather_into_tensor = all_gather_into_tensor

if "reduce_scatter_tensor" not in dir(torch.distributed):
    def reduce_scatter_tensor(output_tensor, input_tensor, group=None):
        world_size = torch.distributed.get_world_size(group)
        chunk_size = input_tensor.numel() // world_size
        input_tensors = input_tensor.chunk(world_size, dim=0)
        reduced_tensor = torch.empty_like(input_tensors[0])
        torch.distributed.reduce_scatter(reduced_tensor, input_tensors, group=group)
        output_tensor.copy_(reduced_tensor)
    torch.distributed.reduce_scatter_tensor = reduce_scatter_tensor

def _reduce(input_: torch.Tensor) -> torch.Tensor:
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_

def _split_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()
    return output

def _split_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size = input_.size(0)
    assert dim_size % world_size == 0
    local_dim_size = dim_size // world_size
    dim_offset = get_tensor_model_parallel_rank() * local_dim_size
    output = input_[dim_offset:dim_offset + local_dim_size].contiguous()
    return output

def _gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(
        tensor_list, input_, group=get_tensor_model_parallel_group()
    )
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output

def _gather_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    shape = list(input_.shape)
    shape[0] *= world_size
    output = torch.empty(shape, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output,
        input_.contiguous(),
        group=get_tensor_model_parallel_group()
    )
    return output

def _reduce_scatter_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    shape = list(input_.shape)
    assert shape[0] % world_size == 0
    shape[0] //= world_size
    output = torch.empty(shape, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(
        output,
        input_.contiguous(),
        group=get_tensor_model_parallel_group()
    )
    return output

class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class _ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)

class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)

class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)

class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, to_model_parallel: bool = True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, to_model_parallel: bool = True):
        ctx.to_model_parallel = to_model_parallel
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.to_model_parallel:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None

class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)

def copy_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)

def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)

def gather_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)

def scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_)

def gather_from_sequence_parallel_region(input_: torch.Tensor, to_model_parallel: bool = True) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(input_, to_model_parallel)

def reduce_scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(input_)

__all__ = [
    "copy_to_tensor_model_parallel_region",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "scatter_to_sequence_parallel_region",
    "gather_from_sequence_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
]
