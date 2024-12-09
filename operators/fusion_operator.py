from operators.base_operator import BaseOperator
import torch
import torch.nn as nn

class ConvBnFusion(BaseOperator):
    """Conv 和 BatchNorm 融合算子"""
    def __init__(self):
        super().__init__("Conv-BN Fusion")

    def apply(self, conv_layer, bn_layer):
        """将卷积层和批归一化层融合"""
        fused_weights = conv_layer.weight * bn_layer.weight.view(-1, 1, 1, 1)
        fused_bias = bn_layer.bias + (conv_layer.bias if conv_layer.bias is not None else 0)
        return fused_weights, fused_bias
