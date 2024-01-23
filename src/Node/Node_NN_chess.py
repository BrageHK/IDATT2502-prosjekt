import numpy as np
import copy
import torch

class NodeNNChess:
    def __init__(self, env, parent=None, action_taken=None, priority=0):
        self.children = []
        self.parent = parent
        self.action_taken = action_taken
        self.env = env
        self.reward = 0
        self.visits = 0
        self.priority = priority

        self.c = 4  # Exploration parameter

    def is_fully_expanded(self):
        return len(self.children) > 0

    def calculate_PUCT(self, child):
        # Quality-value
        q_value = 0
        if child.visits != 0:
            q_value = 1 - ((child.reward / child.visits) + 1) / 2

        # UCB
        ucb = (
            self.c * child.priority * np.sqrt(np.log(self.visits)) / (1 + child.visits)
        )
        return q_value + ucb

    def select(self):
        node = self
        best_child = None
        while node.is_fully_expanded():
            max_UCB = float("-inf")
            for child in node.children:
                UCB_value = node.calculate_PUCT(child)
                if UCB_value > max_UCB:
                    max_UCB = UCB_value
                    best_child = child
            node = best_child
        return node

    def add_dirichlet_noise(self, policy, alpha=0.3):
        """
        Add Dirichlet noise to the policy vector to encourage exploration.
        `alpha` is the parameter for the Dirichlet distribution.
        Assumes that illegal moves have a probability of 0 in the policy vector.
        """
        # Identify legal moves (non-zero probability)
        legal_moves = policy > 0

        # Count the number of legal moves and generate Dirichlet noise on the CPU
        num_legal_moves = legal_moves.sum().item()
        noise = np.random.dirichlet([alpha] * num_legal_moves)

        # Convert the noise to a PyTorch tensor and transfer it to the same device as the policy tensor
        noise_tensor = torch.tensor(noise, device=policy.device, dtype=policy.dtype)

        # Create a tensor to hold the noisy policy, initially a copy of the original policy
        noisy_policy = policy.clone()

        # Apply noise only to legal moves
        noisy_policy[legal_moves] = (1 - 0.25) * noisy_policy[legal_moves] + 0.25 * noise_tensor

        # Return the noisy policy
        return noisy_policy

    def expand(
        self, policy, training=True
    ):  # TODO: make it easier to change training mode
        if self.parent is None and training:
            policy = self.add_dirichlet_noise(policy)

        if self.visits > 0:
            # print("policies: ", policy)
            # print("legal moves from here: ", self.env.legal_actions)
            legal_actions = self.env.legal_actions
            for action, probability in enumerate(policy):
                if probability > 0:
                    if action in legal_actions:
                        # print("legal move ", action)
                        next_env = copy.deepcopy(self.env)
                        next_env.step(action)

                        child = NodeNNChess(
                            next_env, self, action, priority=probability
                        )
                        self.children.append(child)
                    else:
                        # Getting here whould mean logical error
                        print("Illegal move:", action)
                        exit()

    def backpropagate(self, reward):
        self.reward += reward
        self.visits += 1

        if self.parent:
            self.parent.backpropagate(-reward)
