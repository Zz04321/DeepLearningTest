from model_generator.model_builder import ModelBuilder
from differential_test.tester import DifferentialTester
from datasets.data_loader import DataLoader

def main():
    # 1. 初始化工具
    model_builder = ModelBuilder()
    tester = DifferentialTester()
    data_loader = DataLoader()

    # 2. 生成等价模型
    original_model = model_builder.build_model(use_fusion=False)
    fused_model = model_builder.build_model(use_fusion=True)

    # 3. 加载测试数据
    test_data = data_loader.generate_random_data()

    # 4. 执行差分测试
    results = tester.test_models([original_model, fused_model], test_data)
    anomalies = tester.analyze_results()

    # 5. 打印结果
    print("差分测试结果：", results)
    print("异常行为：", anomalies)

if __name__ == "__main__":
    main()
