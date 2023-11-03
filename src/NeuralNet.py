import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

    def forward(self, x):
        return nn.ReLU()(x + self.conv_blocks(x))
    
"""
    Side 11 på: "Improving Monte Carlo Tree Search with Artificial Neural Networks without Heuristics"
"""
class AlphaPredictorNerualNet(nn.Module):
    def __init__(self, num_resBlocks, object_dim=42, action_dim=7, outcome_dim=3, hidden_dim=100, hidden_dim2=32):
        super(AlphaPredictorNerualNet, self).__init__()
        
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
        
        # Predicts probability for the expected outcomes (win current player, draw, win other player)
        # If one of the values in the array is over the threshold - stop simulation, else continue simulation
        #self.value_network = nn.Sequential(
        #    nn.Linear(object_dim, hidden_dim),
        #    nn.Sigmoid(),
        #    nn.Linear(hidden_dim, outcome_dim),
        #    nn.Sigmoid()
        #)

    # Cross Entropy loss
    def loss(self, policy_logits, value_logits, policy_target, value_target): #
        policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_target)
        value_loss = nn.MSELoss()(value_logits, value_target)
        return policy_loss + value_loss
    
    def optimize(self, model, train_dataloader, epoch=20, learning_rate=0.001): #
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        for _ in range(epoch):
            for batch_x, batch_y in train_dataloader:
                policy_target, value_target = batch_y[:, :-1], batch_y[:, -1].unsqueeze(-1)
                policy_logits, value_output = model(batch_x)
                
                # Compute loss gradients
                loss = model.loss(policy_logits, value_output, policy_target, value_target)
                loss.backward()
                
                # Perform optimization by adjusting weights and bias,
                optimizer.step()  
                # Clear gradients for next step
                optimizer.zero_grad() 

                
    # Takes in a state and uses the neuron network to get a policy and a value
    def forward(self, x):
        x = self.initial_network(x)
        for residual_block in self.residual_network:
            x = residual_block(x)
        policy = self.policy_network(x)
        value = self.value_network(x)
        
        return policy, value