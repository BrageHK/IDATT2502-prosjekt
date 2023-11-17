import numpy as np
import time
import torch

from Node.Node import Node
from Node.Node_normalized import NodeNormalized
from Node.Node_NN import NodeNN
from Node.NodeType import NodeType
from Node.Node_threshold import NodeThreshold
from Node.Node_threshold_lightweight import NodeThresholdLightweight

class MCTS:
    def __init__(self, env, num_iterations=69, NODE_TYPE=NodeType.NODE_NORMALIZED, model=None, turn_time=None):
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
            return NodeNN(self.env, state)
        elif self.NODE_TYPE == NodeType.NODE_THRESHOLD:
            return NodeThreshold(env=self.env, state=state, model=self.model)
        elif self.NODE_TYPE == NodeType.NODE_THRESHOLD_LIGHTWEIGHT:
            return NodeThresholdLightweight(env=self.env, state=state, model=self.model)
        else:
            raise Exception("Invalid node type")
        
    @torch.no_grad()
    def get_neural_network_predictions(self, state):
        tensor_state = torch.tensor(self.env.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        policy, value = self.model.forward(tensor_state)
        
        policy = torch.softmax(policy, axis = 1).squeeze(0).detach().cpu().numpy()
        policy_before = policy.copy()
        legal_moves = self.env.get_legal_moves_bool_array(state)
        policy *= legal_moves
        
        sum = np.sum(policy)
        if sum == 0:
            print("Sum is 0!! what")
            policy = legal_moves
            sum = np.sum(policy)
            print("policy before: ")
            print(policy_before)
            print("legal moves: ")
            print(legal_moves)
            if sum == 0:
                raise Exception("No legal moves")
        policy /= sum
        return policy, value.item()
        
    @torch.no_grad()
    def mcts_AlphaZero(self, root):
        # Select
        node = root.select()
        
        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)
        
        if not done:
            # Predicts probabilities for each move and winner probability
            policy, result = self.get_neural_network_predictions(node.state)
                        
            # Expand node
            node.expand(policy)

        # Backpropagate with simulation result
        node.backpropagate(result)

    @torch.no_grad()
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
        
    @torch.no_grad()
    def search(self, state, training=False):
        root = self.create_node(state)
        
        mcts_method = self.mcts if self.NODE_TYPE in [NodeType.NODE, NodeType.NODE_NORMALIZED, NodeType.NODE_THRESHOLD, NodeType.NODE_THRESHOLD_LIGHTWEIGHT] else self.mcts_AlphaZero if self.NODE_TYPE == NodeType.NODE_NN else None
        if mcts_method is None:
            raise Exception("Invalid node type")

        if self.turn_time is None:
            for _ in range(self.num_iterations):
                mcts_method(root)
        else:
            start = time.time()
            while time.time() - start < self.turn_time:
                mcts_method(root)

        visits = np.zeros(self.env.action_space)
        for child in root.children:
            visits[child.action_taken] = child.visits
        #print("Visits: ", visits)
        #print("Best action: ", best_action)
        probabilties = visits / np.sum(visits)
        #print("Probabilties: ", probabilties)
        
        best_action = np.argmax(visits)
        #print("Best action: ", best_action)
        if training:
            return probabilties, best_action
        return best_action