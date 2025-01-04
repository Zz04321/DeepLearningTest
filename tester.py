import torch

class DifferentialTester:
    def test_models(self, models, test_data):
        predictions = []
        for model in models:
            with torch.no_grad():
                predictions.append(model(test_data).cpu().numpy())
        return predictions