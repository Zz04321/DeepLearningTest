import torch

class DataLoader:
    def generate_random_data(self):
        return torch.randn(1, 3, 224, 224)