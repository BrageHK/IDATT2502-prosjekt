import torch
import torch.nn as nn

class NeuralNetThresholdLightweight(nn.Module):
    def __init__(self, env, hidden_dim=100, outcome_dim=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(NeuralNetThresholdLightweight, self).__init__()
        self.device = device
        
        # Predicts probability for the expected outcomes (win current player, draw, win other player)
        # If one of the values in the array is over the threshold - stop simulation, else continue simulation
        self.value_network = nn.Sequential(
            nn.Linear(env.ROW_COUNT * env.COLUMN_COUNT, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, outcome_dim),
            nn.Sigmoid()
        )
        
        self.value_loss_history = []
        
        self.to(device)
        
    def forward(self, x):
        x = self.value_network(x)
        return x