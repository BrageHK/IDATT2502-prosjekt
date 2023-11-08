import numpy as np
import copy
from Connect_four_env import ConnectFour
from NeuralNet import AlphaPredictorNerualNet
import torch


class NodeNN():
    def __init__(self, priority=0, parent=None, env=ConnectFour(), action=None, model=AlphaPredictorNerualNet(4)):
        self.children = []
        self.parent = parent
        self.action = action
        self.env = env
        self.reward = 0
        self.visits = 0
        self.nn_model = model
        self.priority = priority
        
        self.c = 4 # Exploration parameter
        
    def calculate_PUCT(self, child): 
        # Quality-value
        q_value = 0
        if child.visits != 0:
            q_value = 1 - ((child.reward / child.visits) + 1) / 2
            
        # UCB
        ucb = self.c * child.priority * np.sqrt(np.log(self.visits)) / (1 + child.visits) # TODO: In case of errors try with log inside the square root
        
        return q_value + ucb
        

    def select(self):
        node = self
        while bool(node.children): 
            UCB_values = []
            for child in node.children:
                UCB_values.append(node.calculate_PUCT(child))
            node = node.children[np.argmax(UCB_values)]
        
        return node
    
    @torch.no_grad()
    def get_neural_network_predictions(self):
        tensor_state = torch.tensor(self.env.get_encoded_state()).unsqueeze(0)
        policy, value = self.nn_model.forward(tensor_state)
        
        #print("Policy before: ", policy)
        policy = torch.softmax(policy, axis = 1).squeeze(0).detach().numpy()
        policy *= self.env.get_legal_moves_bool_array()
        sum = np.sum(policy)
        if sum != 0:
            policy /= sum
        else:
            print("What is going on? Policy before: ", policy, " sum: ", sum)
            policy = np.zeros(self.env.COLUMN_COUNT)
            policy[np.random.choice(self.env.get_legal_moves())] = 1
        #print("Policy after: ", policy)
        
        value = value.item()
        
        print("Legal moves: ", self.env.get_legal_moves())
        print("policy", policy)
    
        return policy, value

    # Adds a child node for untried action and returns it
    def expand(self, policy):
        if self.visits > 0:
            for action, probability in enumerate(policy):
                if probability > 0: 
                    next_env = self.env.deepcopy()
                    next_env.step(action)
                    child = NodeNN(probability, parent=self, env=next_env, action=action, model=self.nn_model)
                    
                    self.children.append(child)

    def random_act(self, env):
        return np.random.choice(env.get_legal_moves())
    
    def backpropagate(self, value):
        self.visits += 1
        # TODO: invert here???
        self.reward += value

        if self.parent:
            self.parent.backpropagate(-value) # Minus because next backpropocation is for other player
