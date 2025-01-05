import random
import torch
import numpy as np
import torch.nn
import os

from model_builder import ModelBuilder
from tester import DifferentialTester
from data_loader import DataLoader
from operators.fusion_operator import ConvBnFusion, AddMulFusionOperator, MatMulAddFusionOperator, \
    ReluDropoutFusionOperator, PoolingFcFusionOperator
from utils.visualizer import visualize_difference, plot_differences_box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = ["model1", "model2", "model3", "model4", "model5"]
model_creation_methods = [getattr(ModelBuilder(), m) for m in dir(ModelBuilder) if
                          callable(getattr(ModelBuilder(), m)) and m.startswith('build_model')]

# 确保日志目录存在
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

def save_model(model, path):
    # 保存模型的状态字典
    torch.save(model.state_dict(), path)


def load_model(path, model_class, use_fusion):
    # 加载模型，并确保在评估模式
    model = model_class(use_fusion=use_fusion).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # 确保模型处于评估模式
    return model


def set_seed(seed):
    # 设置随机种子，确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def apply_fusion(model, fusion_operators):
    # 在模型中应用融合操作
    for i, module in enumerate(model):
        if i + 1 >= len(model):
            break

        next_module = model[i + 1]

        # 跳过 Flatten 层
        if isinstance(module, torch.nn.Flatten):
            continue

        for fusion_operator in fusion_operators:
            if isinstance(module, fusion_operator.layer_type):
                try:
                    # 尝试应用融合操作
                    fused_weight, fused_bias = fusion_operator.apply(module, next_module)
                    if fused_weight is not None and fused_bias is not None:
                        module.weight.data = fused_weight
                        if hasattr(module, 'bias'):
                            module.bias.data = fused_bias
                except AttributeError:
                    continue
    return model


def get_next_module(model, current_name):
    # 获取当前模块后面的下一个模块
    found = False
    for name, module in model.named_modules():
        if found:
            return module
        if name == current_name:
            found = True
    return None


def main():
    set_seed(42)
    tester = DifferentialTester()
    data_loader = DataLoader()
    fusion_operators = [ConvBnFusion(), AddMulFusionOperator(), MatMulAddFusionOperator(), ReluDropoutFusionOperator(),
                        PoolingFcFusionOperator()]

    # 遍历模型列表
    for i in range(min(len(model_list), len(model_creation_methods))):
        log_file = f'{log_dir}/log_{i + 1}.txt'
        with open(log_file, 'w') as f:
            # 创建并保存 PyTorch 模型
            model = model_creation_methods[i](use_fusion=False).to(device)
            model_save_path = model_list[i]  # 将模型保存在 'model' 文件夹下
            save_model(model, model_save_path)

            # 生成测试数据
            test_data = data_loader.generate_random_data(num_samples=10).to(device)

            # 执行差异测试
            non_fused_model = load_model(model_list[i], model_creation_methods[i], use_fusion=False)
            fused_model = load_model(model_list[i], model_creation_methods[i], use_fusion=True)
            fused_model = apply_fusion(fused_model, fusion_operators)
            non_fused_pred, fused_pred = tester.test_models([non_fused_model, fused_model], test_data)

            differences = []
            for j in range(len(non_fused_pred)):
                # 计算每个预测结果的差异
                difference = np.max(np.abs(non_fused_pred[j] - fused_pred[j]))
                differences.append(difference)
                line = f"{j + 1} - Non-Fused: {non_fused_pred[j]}, Fused: {fused_pred[j]}, Difference: {difference}\n"
                print(line)
                f.write(line)

            # 可视化差异
            visualize_difference(differences, i)

            # 绘制差异的盒须图
            plot_differences_box(differences, i)

            # 检查两个模型的输出是否一致
            consistent = np.allclose(non_fused_pred, fused_pred, atol=1e-5)
            if consistent:
                result = "两个模型的预测结果一致。\n"
            else:
                result = "两个模型的预测结果不一致。\n"
            print(result)
            f.write(result)


if __name__ == "__main__":
    main()
