import torch.nn as nn

class ModelBuilder:
    def build_model_one(self, use_fusion=False):
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        return model

    def build_model_two(self, use_fusion=False):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 112 * 112, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        return model

    def build_model_three(self, use_fusion=False):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 112 * 112, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        return model

    def build_model_four(self, use_fusion=False):
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 112 * 112, 8),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        return model

    def build_model_five(self, use_fusion=False):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 112 * 112, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        return model