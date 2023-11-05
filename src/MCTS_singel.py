import numpy as np
import copy
from Connect_four_env import ConnectFour

class Node():
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
                next_env = copy.deepcopy(self.env)
                next_env.step(action)
                child = Node(parent=self, env=next_env, action=action)
                
                self.children.append(child)
            
    def random_act(self, env):
        return np.random.choice(env.get_legal_moves())

    # Simulate random actions until a terminal state is reached
    def simulate(self):
        simulated_game = copy.deepcopy(self.env)
        current_player = simulated_game.get_player() # -1 or 1
        reward, done = simulated_game.check_game_over(current_player)
        
        while not done:
            action = self.random_act(simulated_game)
            _, reward, done = simulated_game.step(action)

        winning_player = simulated_game.get_player()
        return reward * winning_player, simulated_game.board # Return 1 if the player won, -1 if the player lost, and 0 if it was a draw.
    
    def backpropagate(self, result):
        self.visits += 1
        self.reward += result

        if self.parent:
            self.parent.backpropagate(result)

class MCTS():
    def get_action(self, env, n_simulations=10_000, invert=True, verbose=False):
        if invert: # Invert board from player to AI
            env.board = -env.board
            
        print("env from mcts start ", env.board)
        
        root = Node(env=env)
        
        for _ in range(n_simulations):
            # Select
            node = root.select()
            
            # Expand node
            node.expand()
            
            # Simulate root node
            result, _ = node.simulate()

            # Backpropagate with simulation result
            node.backpropagate(result)
                
        visits = np.array([child.visits for child in root.children])
        if verbose:
            print(visits)
        return root.children[np.argmax(visits)].action