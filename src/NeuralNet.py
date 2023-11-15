import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x += residual
        x = F.relu(x)
        return x
  
class AlphaPredictorNerualNet(nn.Module):
    def __init__(self, num_resBlocks, device, env, hidden_dim=100, hidden_dim_policy=32):
        super(AlphaPredictorNerualNet, self).__init__()
        
        self.device = device
        
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
            nn.Conv2d(hidden_dim, hidden_dim_policy, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim_policy),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim_policy * env.ROW_COUNT * env.COLUMN_COUNT, env.action_space)
        )
        
        # Predicts probability for the current player to win - if over a threshold not simulated, simulated only if below threshold
        self.value_network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim_policy, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim_policy),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim_policy * env.ROW_COUNT * env.COLUMN_COUNT, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    # Takes in a state and uses the neuron network to get a policy and a value
    def forward(self, x):
        x = self.initial_network(x)
        for residual_block in self.residual_network:
            x = residual_block(x)
        policy = self.policy_network(x)
        value = self.value_network(x)
        
        return policy, value
    