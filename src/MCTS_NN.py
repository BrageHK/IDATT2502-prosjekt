import numpy as np
import copy
import torch
from Connect_four_env import ConnectFour
from NeuralNet import AlphaPredictorNerualNet

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
        
    def calculate_PUCT(self, child): # TODO: use alfazeros ucb method
        # Q-value
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
        
        policy = torch.softmax(policy, axis = 1).squeeze(0).detach().numpy()
        policy *= self.env.get_legal_moves_bool_array()
        policy /= np.sum(policy)
        
        value = value.item()
        
        #print("Policy: ", policy)
        #print("Value: ", value)
    
        return policy, value

    # Adds a child node for untried action and returns it
    def expand(self, policy):
        if self.visits > 0:
            for action, probability in enumerate(policy):
                if probability > 0: 
                    next_env = copy.deepcopy(self.env)
                    next_env.step(action)
                    child = NodeNN(probability, parent=self, env=next_env, action=action, model=self.nn_model)
                    
                    self.children.append(child)
            
    def random_act(self, env):
        return np.random.choice(env.get_legal_moves())

    # Simulate random actions until a terminal state is reached
    """ def simulate(self):
        simulated_game = copy.deepcopy(self.env)
        current_player = simulated_game.get_player() # -1 or 1
        reward, done = simulated_game.check_game_over(current_player)
        
        while not done:
            action = self.random_act(simulated_game)
            _, reward, done = simulated_game.step(action)

        winning_player = simulated_game.get_player()
        return reward * winning_player, simulated_game.board # Return 1 if the player won, -1 if the player lost, and 0 if it was a draw. """
    
    def backpropagate(self, value):
        self.visits += 1
        self.reward += value

        if self.parent:
            self.parent.backpropagate(-value) # Minus because next backpropocation is for other player

class MCTSNN():
    def __init__(self, model=ConnectFour()):
        self.model = model
    
    def get_action(self, env, n_simulations=100_000, invert=True, verbose=False, training_return=False):
        if invert: # Invert board from player to AI
            env.board = -env.board
        
        root = NodeNN(env=env, model=self.model)
        
        for _ in range(n_simulations):
            # Select
            node = root.select()
            
            # Predicts probabilities for each move and winner probability
            policy, value = node.get_neural_network_predictions()
            
            # Expand node
            node.expand(policy)
            
            # Simulate root node
            # result, _ = node.simulate()

            # Backpropagate with simulation result
            node.backpropagate(value)
                
        visits = np.array([child.visits for child in root.children])
        best_action = root.children[np.argmax(visits)].action
        probabilties = visits / np.sum(visits)
        if verbose:
            print(visits)
        if training_return:
            return probabilties, best_action
        return best_action

if __name__ == "__main__":
    mcts = MCTSNN(verbose=True)