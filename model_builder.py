import torch
import torch.nn as nn
from operators.fusion_operator import ConvBnFusion


class ModelBuilder:
    """等价模型生成工具"""

    def __init__(self):
        self.fusion_operator = ConvBnFusion()

    def build_model(self, use_fusion=False):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        if use_fusion:
            # 获取卷积和BatchNorm层
            conv_layer = model[0]
            bn_layer = model[1]
            fused_weights, fused_bias = self.fusion_operator.apply(conv_layer, bn_layer)

            # 创建融合后的模型
            fused_model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            fused_model[0].weight.data = fused_weights
            fused_model[0].bias.data = fused_bias
            return fused_model
        return model
