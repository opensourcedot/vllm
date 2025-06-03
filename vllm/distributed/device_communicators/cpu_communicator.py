# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

from vllm.frameworks import current_framework
from vllm.frameworks.distributed import ProcessGroup

from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

from .base_device_communicator import DeviceCommunicatorBase


class CpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[current_framework.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.dist_module = current_framework.distributed

        if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
            self.dist_module = _CPUSHMDistributed(self)

    def all_reduce(self, input_):
        self.dist_module.all_reduce(input_, group=self.device_group)
        return input_

    def gather(self,
               input_: current_framework.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[current_framework.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [current_framework.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None

        # Gather.
        self.dist_module.gather(input_,
                                gather_list,
                                dst=self.ranks[dst],
                                group=self.device_group)

        if self.rank_in_group == dst:
            output_tensor = current_framework.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gather(self, input_: current_framework.Tensor, dim: int = -1) -> current_framework.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # current_framework.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = current_framework.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        self.dist_module.all_gather_into_tensor(output_tensor,
                                                input_,
                                                group=self.device_group)

        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor


class _CPUSHMDistributed:

    def __init__(self, communicator: CpuCommunicator):
        instance_identifier = os.environ["VLLM_DIST_IDENT"]
        self.communicator = communicator

        group_ranks = [str(rank) for rank in self.communicator.ranks]
        shm_group_identifier = f"[{'-'.join(group_ranks)}]"
        self.group_name = f"{instance_identifier}-{shm_group_identifier}-cpushm"

        self.handle = self._init_cpu_shm()

    def _init_cpu_shm(self) -> int:
        handle = current_framework.ops._C.init_shm_manager(
            self.group_name,
            self.communicator.world_size,
            self.communicator.rank,
        )
        current_framework.distributed.barrier(self.communicator.device_group)
        current_framework.ops._C.join_shm_manager(
            handle,
            self.group_name,
        )
        current_framework.distributed.barrier(self.communicator.device_group)

        return handle

    def all_reduce(self,
                   input: current_framework.Tensor,
                   group: Optional[ProcessGroup] = None) -> None:
        current_framework.ops._C.shm_allreduce(self.handle, input)

    def gather(self,
               input: current_framework.Tensor,
               gather_list: Optional[List[current_framework.Tensor]],
               dst: int = -1,
               group: Optional[ProcessGroup] = None) -> None:
        # Note: different from the current_framework gather, here we use local dst rank.
        current_framework.ops._C.shm_gather(self.handle, input, gather_list,
                                current_framework.distributed.get_group_rank(group, dst))

    def all_gather_into_tensor(self,
                               output: current_framework.Tensor,
                               input: current_framework.Tensor,
                               group: Optional[ProcessGroup] = None) -> None:
        current_framework.ops._C.shm_all_gather(self.handle, input, output)
