import torch
import torch.nn as nn

# Model architecture (must match the training code)
class DivorcePredictor(nn.Module):
    def __init__(self, input_dim):
        super(DivorcePredictor, self).__init__()
        # why do we need (two) hidden layers-- what is the logic here
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), # why output is 32
            nn.ReLU(),
            nn.Linear(32, 16), # why output is 16
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
