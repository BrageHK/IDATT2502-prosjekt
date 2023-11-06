import numpy as np

from Node_R import Node
from NodeType import NodeType

class MCTS:
    def __init__(self, env, num_iterations, NODE_TYPE=NodeType.NODE_SINGLE, model=None):
        self.env = env
        self.num_iterations = num_iterations
        self.NODE_TYPE = NODE_TYPE
        self.model = model
        
        
    def create_node(self, state):
        if self.NODE_TYPE == NodeType.NODE_SINGLE:
            return Node(self.env, state)
        # elif self.NODE_TYPE == NodeType.NODE_DOUBLE:
        #     return NodeDouble(self.env, state)
        # elif self.NODE_TYPE == NodeType.NODE_NN:
        #     return NodeNN(self.env, self)
        
    def mcts_AlphaZero(self, root):
        # Select
        node = root.select()
        
        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)
        
        if not done:
            # Predicts probabilities for each move and winner probability
            policy, value = node.get_neural_network_predictions()
            
            # Expand node
            node.expand(policy)

        # Backpropagate with simulation result
        node.backpropagate(value)
    
    def mcts(self, root):
        # Select
        node = root.select()
        
        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)
        
        if not done:
            # Expand node
            node = node.expand()
            # Simulate root node
            result, _ = node.simulate()
                
        # Backpropagate with simulation result
        node.backpropagate(result)
        
    def search(self, state, training=False):
        root = self.create_node(state)
        
        if self.NODE_TYPE == NodeType.NODE_SINGLE or self.NODE_TYPE == NodeType.NODE_DOUBLE:
            for _ in range(self.num_iterations):
                self.mcts(root)
        elif self.NODE_TYPE == NodeType.NODE_NN:
            for _ in range(self.num_iterations):
                self.mcts_AlphaZero(root)
        else:
            raise Exception("Invalid node type")
            
        if training:
            visits = np.zeros(self.env.COLUMN_COUNT)
            for child in root.children:
                visits[child.action] = child.visits
            probabilties = visits / np.sum(visits)
            return probabilties, best_action
        
        visits = np.array([child.visits for child in root.children])
        print("Visits: ", visits)
        best_action = max(root.children, key=lambda child: child.visits, default=None).action_taken
        return best_action