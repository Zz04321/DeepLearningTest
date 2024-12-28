import torch
import torch.nn as nn

class BaseOperator:
    """基础算子类"""
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        raise NotImplementedError("子类需要实现 apply 方法")
