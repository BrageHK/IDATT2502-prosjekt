import torch
import torch.nn as nn
import time

class NeuralNetThreshold(nn.Module):
    def __init__(self, env, object_dim=3, hidden_dim=100, outcome_dim=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(NeuralNetThreshold, self).__init__()
        self.device = device
        
        # Predicts probability for the expected outcomes (win current player, draw, win other player)
        # If one of the values in the array is over the threshold - stop simulation, else continue simulation
        self.value_network = nn.Sequential(
            nn.Conv2d(object_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(hidden_dim * env.ROW_COUNT * env.COLUMN_COUNT, outcome_dim),
            nn.Sigmoid()
        )
        
        self.value_loss_history = []
        
        self.to(device)
        
    def forward(self, x):
        x = self.value_network(x)
        return x