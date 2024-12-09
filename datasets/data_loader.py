import torch
import numpy as np

class DataLoader:
    """数据加载工具"""
    @staticmethod
    def generate_random_data(num_samples=100, input_shape=(3, 64, 64)):
        """生成随机输入数据"""
        return torch.rand(num_samples, *input_shape)
