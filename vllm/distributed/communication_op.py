# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union

from vllm.frameworks import current_framework
import vllm.frameworks.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: current_framework.Tensor) -> current_framework.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: current_framework.Tensor,
                                     dim: int = -1) -> current_framework.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: current_framework.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[current_framework.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[current_framework.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not current_framework.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
