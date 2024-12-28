from operators.base_operator import BaseOperator
import torch
import torch.nn as nn

class BaseOperator:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

class ConvBnFusion(BaseOperator):
    """Conv 和 BatchNorm 融合算子"""
    def __init__(self):
        super().__init__("Conv-BN Fusion")

    def apply(self, conv_layer, bn_layer):
        """将卷积层和批归一化层融合"""
        # 获取卷积层和批归一化层的参数
        conv_weight = conv_layer.weight
        conv_bias = conv_layer.bias if conv_layer.bias is not None else torch.zeros(conv_layer.out_channels)
        bn_weight = bn_layer.weight
        bn_bias = bn_layer.bias
        bn_mean = bn_layer.running_mean
        bn_var = bn_layer.running_var
        bn_eps = bn_layer.eps

        # 计算融合后的权重和偏置
        fused_weight = bn_weight.view(-1, 1, 1, 1) * conv_weight / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)
        fused_bias = bn_weight * (conv_bias - bn_mean) / torch.sqrt(bn_var + bn_eps) + bn_bias

        # 返回融合后的权重和偏置
        return fused_weight, fused_bias
