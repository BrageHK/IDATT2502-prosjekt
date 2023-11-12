import numpy as np
import time

from Node.Node import Node
from Node.Node_normalized import NodeNormalized
from Node.Node_NN import NodeNN
from Node.NodeType import NodeType
from Node.Node_threshold import NodeThreshold

class MCTS:
    def __init__(self, env, num_iterations, NODE_TYPE=NodeType.NODE_NORMALIZED, model=None, turn_time=None):
        self.env = env
        self.num_iterations = num_iterations
        self.NODE_TYPE = NODE_TYPE
        self.model = model
        self.turn_time = turn_time
        
    def create_node(self, state):
        if self.NODE_TYPE == NodeType.NODE:
            return Node(self.env, state)
        elif self.NODE_TYPE == NodeType.NODE_NORMALIZED:
            return NodeNormalized(self.env, state)
        elif self.NODE_TYPE == NodeType.NODE_NN:
            return NodeNN(self.env, state, model=self.model)
        elif self.NODE_TYPE == NodeType.NODE_THRESHOLD:
            return NodeThreshold(self.env, state)
        else:
            raise Exception("Invalid node type")
        
    def mcts_AlphaZero(self, root):
        # Select
        node = root.select()
        
        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)
        
        if not done:
            # Predicts probabilities for each move and winner probability
            policy, result = node.get_neural_network_predictions()
            
            # Expand node
            node.expand(policy)

        # Backpropagate with simulation result
        node.backpropagate(result)
    
    def mcts(self, root):
        # Select
        node = root.select()
        
        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)
        
        if not done:
            # Expand node
            node.expand()
            # Simulate root node
            result, _ = node.simulate()
                
        # Backpropagate with simulation result
        node.backpropagate(result)
        
    def search(self, state, training=False):
        root = self.create_node(state)
        
        if self.turn_time == None:
            if self.NODE_TYPE == NodeType.NODE or self.NODE_TYPE == NodeType.NODE_NORMALIZED or self.NODE_TYPE == NodeType.NODE_THRESHOLD:
                for _ in range(self.num_iterations):
                    self.mcts(root)
            elif self.NODE_TYPE == NodeType.NODE_NN:
                for _ in range(self.num_iterations):
                    self.mcts_AlphaZero(root)
            else:
                raise Exception("Invalid node type")
        else:
            if self.NODE_TYPE == NodeType.NODE or self.NODE_TYPE == NodeType.NODE_NORMALIZED or self.NODE_TYPE == NodeType.NODE_THRESHOLD:
                start = time.time()
                while time.time() - start < self.turn_time:
                    self.mcts(root)
            elif self.NODE_TYPE == NodeType.NODE_NN:
                start = time.time()
                while time.time() - start < self.turn_time:
                    self.mcts_AlphaZero(root)
            else:
                raise Exception("Invalid node type")
            
        visits = np.array([child.visits for child in root.children])
        #print("Visits: ", visits)
        best_action = max(root.children, key=lambda child: child.visits, default=None).action_taken
        #print("Best action: ", best_action)
        
        if training:
            visits = np.zeros(self.env.COLUMN_COUNT)
            for child in root.children:
                visits[child.action_taken] = child.visits
            probabilties = visits / np.sum(visits)
            return probabilties, best_action
        return best_action