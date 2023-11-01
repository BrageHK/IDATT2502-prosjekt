import torch
import torch.nn as nn

"""
    Side 11 på: "Improving Monte Carlo Tree Search with Artificial Neural Networks without Heuristics"
"""
class PredictorNerualNet(nn.Module):
    def __init__(self, object_dim, action_dim=3, hidden_dim=100):
        super(PredictorNerualNet, self).__init__()
        
        self.lin1 = nn.Linear(object_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.activation(x)
        x = self.lin3(x)
        return x