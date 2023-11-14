import torch
import torch.nn as nn
import random
import numpy as np
import pickle
import torch.nn.functional as F

    
class ResBlock(nn.Module):
  def __init__(self, num_hidden):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_hidden),
        nn.ReLU(),
        nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_hidden)
    )

  def forward(self, x):
      residual = x
      x = self.block(x)
      x += residual
      x = F.relu(x)
      return x
    
"""
    Side 11 på: "Improving Monte Carlo Tree Search with Artificial Neural Networks without Heuristics"
"""
class AlphaPredictorNerualNet(nn.Module):
    def __init__(self, num_resBlocks, object_dim=42, action_dim=7, hidden_dim=100, hidden_dim2=32):
        super(AlphaPredictorNerualNet, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_loss_history = []
        self.value_loss_history = []
        
        self.optimizer = None
        
        self.initial_network = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1), # [moves current player, available moves, moves by oponent]
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Residual network
        self.residual_network = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(num_resBlocks)]
        )

        # Adjust the policy to towards the goale of winning the game - retunres a probability distributions of leagal moves
        self.policy_network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim2 * object_dim, action_dim)
        )
        
        # Predicts probability for the current player to win - if over a threshold not simulated, simulated only if below threshold
        self.value_network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim2 * object_dim, 1),
            nn.Tanh()
        )
        
        self.to(self.device)
        
    def save_loss_values_to_file(self, filename):
        with open(filename, "wb") as file:
            pickle.dump((self.policy_loss_history, self.value_loss_history), file)

    # Cross Entropy loss
    def loss(self, policy_logits, value_logits, policy_target, value_target): #
        policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_target)
        value_loss = nn.MSELoss()(value_logits, value_target)
        self.value_loss_history.append(value_loss.item())
        self.policy_loss_history.append(policy_loss.item())
        return policy_loss + value_loss
            
    def optimize(self, model, memory, epoch=600, learning_rate=0.001, batch_size=128):
        random.shuffle(memory)
            
        if model.optimizer is None:
            model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
             

        for i in range(epoch):
            if i+1 % 10 == 0:
                    print("Starting epoch: ", i+1)
            for batch_index in range(0, len(memory), batch_size):
                sample = memory[batch_index:min(len(memory) - 1, batch_index + batch_size)] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
                states, policy_targets, value_targets = zip(*sample)
                
                states = np.array(states)
                policy_targets = np.array(policy_targets)
                value_targets = np.array([np.array(item).reshape(-1, 1) for item in value_targets])
            
                states = torch.tensor(states, dtype=torch.float32, device=model.device)
                policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=model.device)
                value_targets = torch.tensor(value_targets, dtype=torch.float32, device=model.device)
                
                policy_output, value_output = model(states)
                
                loss = model.loss(policy_output, value_output, policy_targets, value_targets.squeeze(-1))

                loss.backward()
                model.optimizer.step()
                model.optimizer.zero_grad()

                # Calculate average loss for the epoch and append to history
    
    # Takes in a state and uses the neuron network to get a policy and a value
    def forward(self, x):
        x = self.initial_network(x)
        for residual_block in self.residual_network:
            x = residual_block(x)
        policy = self.policy_network(x)
        value = self.value_network(x)
        
        return policy, value
    