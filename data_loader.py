import torch

class DataLoader:
    def generate_random_data(self, num_samples=1):
        return torch.randn(num_samples, 3, 224, 224)