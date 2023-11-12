import torch
import torch.nn as nn
import random
import numpy as np
import pickle

class NeuralNetThreshold(nn.Module):
    def __init__(self, object_dim=3, hidden_dim=100, outcome_dim=3):
        super(NeuralNetThreshold, self).__init__()
        # Predicts probability for the expected outcomes (win current player, draw, win other player)
        # If one of the values in the array is over the threshold - stop simulation, else continue simulation
        self.value_network = nn.Sequential(
            nn.Conv2d(object_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(hidden_dim * 42, outcome_dim),
            nn.Sigmoid()
        )
        
        self.value_loss_history = []
        
        
        
    def forward(self, x):
        return self.value_network(x)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        
    def save_loss_values_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.value_loss_history, f)
        
    def loss(self, value, expected_value):
        value_loss = nn.MSELoss()(value, expected_value)
        self.value_loss_history.append(value_loss.item())
        return value_loss
    
    def optimize(self, model, memory, epoch=20, learning_rate=0.001, batch_size=64):
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
        for _ in range(epoch):
            random.shuffle(memory)
            
            for batch_index in range(0, len(memory), batch_size):
                end_index = min(len(memory) - 1, batch_index + batch_size)
                batch = memory[batch_index:end_index]

                states, value_targets = zip(*batch)
                # print("Value targets")
                # print(value_targets)
                states = np.array(states)
                value_targets = np.array([np.array(item).reshape(-1, 1) for item in value_targets])
            
                states = torch.tensor(states, dtype=torch.float32)
                value_targets = torch.tensor(value_targets, dtype=torch.float32)

                predicted_values = model(states)
                
                # Compute loss gradients
                model.loss(predicted_values, value_targets, value_targets.squeeze(-1)).backward()
                # Perform optimization by adjusting weights and bias,
                self.optimizer.step()
                # Clear gradients for next step
                self.optimizer.zero_grad()