import random
import torch
import numpy as np
import torch.nn

from model_builder import ModelBuilder
from tester import DifferentialTester
from datasets.data_loader import DataLoader
from operators.fusion_operator import ConvBnFusion, AddMulFusionOperator, MatMulAddFusionOperator, ReluDropoutFusionOperator, PoolingFcFusionOperator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = ["model1", "model2", "model3", "model4", "model5"]
model_creation_methods = [getattr(ModelBuilder(), m) for m in dir(ModelBuilder) if callable(getattr(ModelBuilder(), m)) and m.startswith('build_model')]

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, model_class, use_fusion):
    model = model_class(use_fusion=use_fusion).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # Ensure the model is in evaluation mode
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# def apply_fusion(model, fusion_operators):
#     for name, module in model.named_modules():
#         for fusion_operator in fusion_operators:
#             if isinstance(module, fusion_operator.layer_type):
#                 next_module = get_next_module(model, name)
#                 if next_module:
#                     fused_weight, fused_bias = fusion_operator.apply(module, next_module)
#                     if hasattr(module, 'weight') and hasattr(module, 'bias'):
#                         module.weight.data.copy_(fused_weight)
#                         module.bias.data.copy_(fused_bias)
#     return model

def apply_fusion(model, fusion_operators):
    for i, module in enumerate(model):
        if i + 1 >= len(model):
            break
            
        next_module = model[i + 1]
        
        # Skip Flatten layers
        if isinstance(module, torch.nn.Flatten):
            continue
            
        for fusion_operator in fusion_operators:
            if isinstance(module, fusion_operator.layer_type):
                try:
                    fused_weight, fused_bias = fusion_operator.apply(module, next_module)
                    if fused_weight is not None and fused_bias is not None:
                        module.weight.data = fused_weight
                        if hasattr(module, 'bias'):
                            module.bias.data = fused_bias
                except AttributeError:
                    continue # Skip if layer doesn't have required attributes
    return model


def get_next_module(model, current_name):
    found = False
    for name, module in model.named_modules():
        if found:
            return module
        if name == current_name:
            found = True
    return None

def main():
    set_seed(42)
    model_builder = ModelBuilder()
    tester = DifferentialTester()
    data_loader = DataLoader()
    fusion_operators = [ConvBnFusion(), AddMulFusionOperator(), MatMulAddFusionOperator(), ReluDropoutFusionOperator(), PoolingFcFusionOperator()]

    for i in range(min(len(model_list), len(model_creation_methods))):
        log_file = f'log_{i + 1}.txt'
        with open(log_file, 'w') as f:
            # Create and save PyTorch model
            model = model_creation_methods[i](use_fusion=False).to(device)
            save_model(model, model_list[i])

            # Generate test data
            test_data = data_loader.generate_random_data().to(device)

            # Perform differential testing
            non_fused_model = load_model(model_list[i], model_creation_methods[i], use_fusion=False)
            fused_model = load_model(model_list[i], model_creation_methods[i], use_fusion=True)
            fused_model = apply_fusion(fused_model, fusion_operators)
            non_fused_pred, fused_pred = tester.test_models([non_fused_model, fused_model], test_data)

            for j in range(len(non_fused_pred)):
                line = f"{j + 1} - Non-Fused: {non_fused_pred[j]}, Fused: {fused_pred[j]}\n"
                print(line)
                f.write(line)

            # Check if the outputs of the two models are consistent
            consistent = np.allclose(non_fused_pred, fused_pred, atol=1e-5)
            if consistent:
                result = "两个模型的预测结果一致。\n"
            else:
                result = "两个模型的预测结果不一致。\n"
            print(result)
            f.write(result)

if __name__ == "__main__":
    main()