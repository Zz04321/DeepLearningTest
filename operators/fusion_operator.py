import torch
import torch.nn as nn
from .base_operator import BaseOperator

class ConvBnFusion(BaseOperator):
    """Conv 和 BatchNorm 融合算子"""
    layer_type = nn.Conv2d

    def __init__(self):
        super().__init__("Conv-BN Fusion")

    def apply(self, conv_layer, bn_layer):

        conv_weight = conv_layer.weight
        conv_bias = conv_layer.bias if conv_layer.bias is not None else torch.zeros(conv_layer.out_channels)
        bn_weight = bn_layer.weight
        bn_bias = bn_layer.bias
        bn_mean = bn_layer.running_mean
        bn_var = bn_layer.running_var
        bn_eps = bn_layer.eps

        fused_weight = bn_weight.view(-1, 1, 1, 1) * conv_weight / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)
        fused_bias = bn_weight * (conv_bias - bn_mean) / torch.sqrt(bn_var + bn_eps) + bn_bias

        return fused_weight, fused_bias

class AddMulFusionOperator(BaseOperator):
    """Add 和 Mul 融合算子"""
    layer_type = nn.Linear

    def __init__(self):
        super().__init__("Add-Mul Fusion")

    def apply(self, add_layer, mul_layer):
        add_weight = add_layer.weight
        add_bias = add_layer.bias if add_layer.bias is not None else torch.zeros(add_layer.out_channels)
        mul_weight = mul_layer.weight
        mul_bias = mul_layer.bias if mul_layer.bias is not None else torch.zeros(mul_layer.out_channels)

        fused_weight = add_weight * mul_weight
        fused_bias = add_bias * mul_weight + mul_bias

        return fused_weight, fused_bias

class MatMulAddFusionOperator(BaseOperator):
    """MatMul 和 Add 融合算子"""
    layer_type = nn.Linear

    def __init__(self):
        super().__init__("MatMul-Add Fusion")

    def apply(self, matmul_layer, add_layer):
        matmul_weight = matmul_layer.weight
        matmul_bias = matmul_layer.bias if matmul_layer.bias is not None else torch.zeros(matmul_layer.out_channels)
        add_weight = add_layer.weight
        add_bias = add_layer.bias if add_layer.bias is not None else torch.zeros(add_layer.out_channels)

        fused_weight = matmul_weight + add_weight
        fused_bias = matmul_bias + add_bias

        return fused_weight, fused_bias

class ReluDropoutFusionOperator(BaseOperator):
    """ReLU 和 Dropout 融合算子"""
    layer_type = nn.ReLU

    def __init__(self):
        super().__init__("ReLU-Dropout Fusion")

    def apply(self, relu_layer, dropout_layer):
        if not isinstance(dropout_layer, nn.Dropout):
            return None, None
        # 创建一个新的融合层，将 ReLU 和 Dropout 的功能整合
        class FusedReluDropout(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p  # Dropout 概率

            def forward(self, x):
                x = torch.relu(x)  # ReLU 激活
                return nn.functional.dropout(x, p=self.p, training=self.training)
        return FusedReluDropout(p=dropout_layer.p), None


class PoolingFcFusionOperator(BaseOperator):
    layer_type = nn.MaxPool2d

    def __init__(self):
        super().__init__("Pooling-FC Fusion")

    def apply(self, pooling_layer, fc_layer):
        # Skip if not correct layer types
        if not isinstance(pooling_layer, (nn.MaxPool2d, nn.AvgPool2d)) or \
           not isinstance(fc_layer, nn.Linear):
            return None, None
            
        # Skip if Flatten layer in between
        if isinstance(fc_layer, nn.Flatten):
            return None, None
            
        return pooling_layer.weight if hasattr(pooling_layer, 'weight') else None, \
               pooling_layer.bias if hasattr(pooling_layer, 'bias') else None