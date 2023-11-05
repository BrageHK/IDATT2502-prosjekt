import copy
import numpy as np
from Connect_four_env import ConnectFour

class NodeSingel():
    def __init__(self, parent=None, env=ConnectFour(), action=None):
        self.children = []
        self.parent = parent
        self.action = action
        self.env = env
        self.reward = 0
        self.visits = 0
        
        self.c = 4 # Exploration parameter
        
    def calculate_UCB(self, child):
        if child.visits == 0:
            return float('inf')
        
        return self.env.get_player() * child.reward / child.visits + np.sqrt(self.c * np.log(self.visits) / child.visits)

    def select(self):
        node = self
        while bool(node.children): 
            UCB_values = []
            for child in node.children:
                UCB_values.append(node.calculate_UCB(child))
            node = node.children[np.argmax(UCB_values)]
        
        return node

    # Adds a child node for untried action and returns it
    def expand(self):
        if self.visits > 0:
            legal_moves = self.env.get_legal_moves()
            for action in legal_moves:
                next_env = self.env.deepcopy()
                next_env.step(action)
                child = NodeSingel(parent=self, env=next_env, action=action)
                
                self.children.append(child)
            
    def random_act(self, env):
        return np.random.choice(env.get_legal_moves())

    # Simulate random actions until a terminal state is reached
    def simulate(self):
        simulated_game = self.env.deepcopy()
        current_player = simulated_game.get_player() # -1 or 1
        reward, done = simulated_game.check_game_over(current_player)
        
        while not done:
            action = self.random_act(simulated_game)
            reward, done = simulated_game.step(action)

        winning_player = simulated_game.get_player()
        return reward * winning_player, simulated_game.board # Return 1 if the player won, -1 if the player lost, and 0 if it was a draw.
    
    def backpropagate(self, result):
        self.visits += 1
        if self.env.is_inverted:
            self.reward -= result
        self.reward += result

        if self.parent:
            self.parent.backpropagate(result)