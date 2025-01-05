import torch
import torch.nn as nn
from .base_operator import BaseOperator


class ConvBnFusion(BaseOperator):
    """Conv 和 BatchNorm 融合算子"""
    layer_type = nn.Conv2d  # 定义适用的层类型为卷积层

    def __init__(self):
        super().__init__("Conv-BN Fusion")  # 调用父类初始化，设置算子名称

    def apply(self, conv_layer, bn_layer):
        """
        应用 Conv 和 BatchNorm 融合
        参数:
            conv_layer: 卷积层
            bn_layer: BatchNorm 层
        返回:
            融合后的权重和偏置
        """
        # 获取卷积层的权重和偏置
        conv_weight = conv_layer.weight
        conv_bias = conv_layer.bias if conv_layer.bias is not None else torch.zeros(conv_layer.out_channels)

        # 获取 BatchNorm 层的参数
        bn_weight = bn_layer.weight
        bn_bias = bn_layer.bias
        bn_mean = bn_layer.running_mean
        bn_var = bn_layer.running_var
        bn_eps = bn_layer.eps

        # 融合后的卷积权重计算
        fused_weight = bn_weight.view(-1, 1, 1, 1) * conv_weight / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)
        # 融合后的卷积偏置计算
        fused_bias = bn_weight * (conv_bias - bn_mean) / torch.sqrt(bn_var + bn_eps) + bn_bias

        return fused_weight, fused_bias


class AddMulFusionOperator(BaseOperator):
    """Add 和 Mul 融合算子"""
    layer_type = nn.Linear  # 定义适用的层类型为全连接层

    def __init__(self):
        super().__init__("Add-Mul Fusion")  # 调用父类初始化，设置算子名称

    def apply(self, add_layer, mul_layer):
        """
        应用 Add 和 Mul 融合
        参数:
            add_layer: 加法层
            mul_layer: 乘法层
        返回:
            融合后的权重和偏置
        """
        # 获取加法层的权重和偏置
        add_weight = add_layer.weight
        add_bias = add_layer.bias if add_layer.bias is not None else torch.zeros(add_layer.out_channels)

        # 获取乘法层的权重和偏置
        mul_weight = mul_layer.weight
        mul_bias = mul_layer.bias if mul_layer.bias is not None else torch.zeros(mul_layer.out_channels)

        # 融合后的加权权重和偏置
        fused_weight = add_weight * mul_weight
        fused_bias = add_bias * mul_weight + mul_bias

        return fused_weight, fused_bias


class MatMulAddFusionOperator(BaseOperator):
    """MatMul 和 Add 融合算子"""
    layer_type = nn.Linear  # 定义适用的层类型为全连接层

    def __init__(self):
        super().__init__("MatMul-Add Fusion")  # 调用父类初始化，设置算子名称

    def apply(self, matmul_layer, add_layer):
        """
        应用 MatMul 和 Add 融合
        参数:
            matmul_layer: 矩阵乘法层
            add_layer: 加法层
        返回:
            融合后的权重和偏置
        """
        # 获取矩阵乘法层的权重和偏置
        matmul_weight = matmul_layer.weight
        matmul_bias = matmul_layer.bias if matmul_layer.bias is not None else torch.zeros(matmul_layer.out_channels)

        # 获取加法层的权重和偏置
        add_weight = add_layer.weight
        add_bias = add_layer.bias if add_layer.bias is not None else torch.zeros(add_layer.out_channels)

        # 融合后的权重和偏置
        fused_weight = matmul_weight + add_weight
        fused_bias = matmul_bias + add_bias

        return fused_weight, fused_bias


class ReluDropoutFusionOperator(BaseOperator):
    """ReLU 和 Dropout 融合算子"""
    layer_type = nn.ReLU  # 定义适用的层类型为 ReLU 激活层

    def __init__(self):
        super().__init__("ReLU-Dropout Fusion")  # 调用父类初始化，设置算子名称

    def apply(self, relu_layer, dropout_layer):
        """
        应用 ReLU 和 Dropout 融合
        参数:
            relu_layer: ReLU 激活层
            dropout_layer: Dropout 层
        返回:
            融合后的层（ReLU 和 Dropout 结合）
        """
        if not isinstance(dropout_layer, nn.Dropout):  # 确保 dropout 层是 nn.Dropout 类型
            return None, None

        # 创建一个新的融合层，将 ReLU 和 Dropout 的功能整合
        class FusedReluDropout(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p  # Dropout 概率

            def forward(self, x):
                x = torch.relu(x)  # 应用 ReLU 激活
                return nn.functional.dropout(x, p=self.p, training=self.training)  # 应用 Dropout

        return FusedReluDropout(p=dropout_layer.p), None  # 返回融合后的层


class PoolingFcFusionOperator(BaseOperator):
    """Pooling 和 FC 融合算子"""
    layer_type = nn.MaxPool2d  # 定义适用的层类型为最大池化层

    def __init__(self):
        super().__init__("Pooling-FC Fusion")  # 调用父类初始化，设置算子名称

    def apply(self, pooling_layer, fc_layer):
        """
        应用 Pooling 和 FC 融合
        参数:
            pooling_layer: 池化层（MaxPool2d 或 AvgPool2d）
            fc_layer: 全连接层（Linear）
        返回:
            融合后的权重和偏置
        """
        # 如果层类型不匹配，返回 None
        if not isinstance(pooling_layer, (nn.MaxPool2d, nn.AvgPool2d)) or \
                not isinstance(fc_layer, nn.Linear):
            return None, None

        # 如果池化层和全连接层之间有 Flatten 层，跳过
        if isinstance(fc_layer, nn.Flatten):
            return None, None

        # 返回池化层的权重和偏置（如果存在）
        return pooling_layer.weight if hasattr(pooling_layer, 'weight') else None, \
            pooling_layer.bias if hasattr(pooling_layer, 'bias') else None
