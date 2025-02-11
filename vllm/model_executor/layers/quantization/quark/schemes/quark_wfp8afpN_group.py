# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter)

__all__ = ["QuarkWfp4aFpNGroup"]


class QuarkWfp4aFpNGroup(QuarkScheme):

    def __init__(self, 
                 weight_qscheme: str, 
                 weight_mx_scale_constraint: Optional[bool],
                 weight_group_size: Optional[int],
                 weight_axis: Optional[int],
                 input_qscheme: Optional[str],
                 input_mx_scale_constraint: Optional[bool],
                 input_group_size: Optional[int],
                 input_axis: Optional[int],
                 input_dtype: Optional[str]):
        self.weight_qscheme = weight_qscheme
        self.weight_mx_scale_constraint = weight_mx_scale_constraint
        self.weight_group_size = weight_group_size
        self.weight_axis = weight_axis
        self.input_qscheme = input_qscheme
        self.input_mx_scale_constraint = input_mx_scale_constraint
        self.input_group_size = input_group_size
        self.input_axis = input_axis
        self.input_dtype = input_dtype

    def process_weights_after_loading(self, layer) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data,
                                       requires_grad=False)

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition//2,
            dtype=torch.uint8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        
        num_groups = input_size_per_partition // self.weight_group_size
        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(data=torch.empty(
            output_partition_sizes,
            num_groups,
            dtype=params_dtype),
                                                input_dim=1,
                                                output_dim=0,
                                                weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
