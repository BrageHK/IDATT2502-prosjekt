import numpy as np
from Node_R import Node

class MCTS:
    def __init__(self, game, num_iterations):
        self.game = game
        self.num_iterations = num_iterations
        
    def search(self, state):
        root = Node(self.game, state)
        
        for _ in range(self.num_iterations):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs