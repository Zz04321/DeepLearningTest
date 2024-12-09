import pytest
import torch
from differential_test.tester import DifferentialTester
from model_generator.model_builder import ModelBuilder
from datasets.data_loader import DataLoader

def test_differential_tester():
    model_builder = ModelBuilder()
    tester = DifferentialTester()
    data_loader = DataLoader()

    # 生成两个等价模型
    original_model = model_builder.build_model(use_fusion=False)
    fused_model = model_builder.build_model(use_fusion=True)

    test_data = data_loader.generate_random_data()

    # 测试模型差异
    results = tester.test_models([original_model, fused_model], test_data)
    anomalies = tester.analyze_results()

    assert len(results) > 0, "没有计算出差异"
    assert len(anomalies) == 0, "检测到异常行为"
