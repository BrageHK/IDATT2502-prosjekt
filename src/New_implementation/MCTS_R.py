import numpy as np
from Node_R import Node

class MCTS:
    def __init__(self, env, num_iterations):
        self.env = env
        self.num_iterations = num_iterations
        
    def search(self, state):
        root = Node(self.env, state)
        
        for _ in range(self.num_iterations):
            node = root
            node = node.select()
            
            reward, done = self.env.check_game_over(node.state, node.action_taken)
            reward = self.env.get_opponent_value(reward)
            
            if not done:
                node = node.expand()
                reward, _ = node.simulate()
                
            node.backpropagate(reward)    
            
            
        action_probs = np.zeros(self.env.action_space)
        for child in root.children:
            action_probs[child.action_taken] = child.visits
        action_probs /= np.sum(action_probs)
        action = np.argmax(action_probs)
        #TODO returnere action_probs eller action?
        return action