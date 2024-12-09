import torch
import numpy as np

class DifferentialTester:
    """差分测试工具"""
    def __init__(self):
        self.results = []

    def test_models(self, models, test_data):
        """对比多个等价模型的输出"""
        outputs = [model(test_data) for model in models]
        for i, out_a in enumerate(outputs):
            for j, out_b in enumerate(outputs):
                if i >= j:
                    continue
                difference = np.max(np.abs(out_a.detach().numpy() - out_b.detach().numpy()))
                self.results.append((i, j, difference))
        return self.results

    def analyze_results(self, threshold=1e-5):
        """分析测试结果，判断是否有异常行为"""
        anomalies = [res for res in self.results if res[2] > threshold]
        return anomalies
