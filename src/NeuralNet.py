import torch
import torch.nn as nn

"""
Dette er bare et nevralnett som er brukt i DQN, vi må sikkert
forandre på det ganske mye.
"""
class QNetwork(nn.Module):
    def __init__(self, object_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.lin1 = nn.Linear(object_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        x = self.lin3(x)
        return x