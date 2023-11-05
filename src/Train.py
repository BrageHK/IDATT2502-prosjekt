from MCTS_NN import MCTSNN
from Connect_four_env import ConnectFour
from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from NeuralNet import AlphaPredictorNerualNet
import copy

class Trainer:
    def __init__(self, model=AlphaPredictorNerualNet(4)):
        self.model = model
        self.mcts = MCTSNN(model)
    
    def save_model_and_optimizer(self, depth, games):
        torch.save(self.model.state_dict(), f"model-{depth}-depth-and-{games}-games.pt")
        torch.save(self.model.optimizer.state_dict(), f"optimizer-{depth}-depth-and-{games}-games.pt")

    def train(self, num_games, nn_depth):
        memory = []
        self.model.eval()
        
        # Plays num_games against itself and stores the data in memory
        for _ in range(num_games):
            memory += self.play_game(nn_depth)
            
        # Trains the model on the data in memory
        self.model.train() # Sets training mode
        
        self.model.optimize(self.model, memory)
        
        torch.save(self.model.state_dict(), "model.pt")
            
    def play_game(self, nn_depth):
        env = ConnectFour()
        memory = []
        invert = False # Player 1 first makes a move
        done = False
        reward = 0
        
        # state = env.get_encoded_state() # TODO: can be the cause of the bug
        
        
        while not done:

            mcts_prob, action = self.mcts.get_action(env=env, n_simulations=nn_depth, training_return=True, invert=True)  # assuming your MCTS has an option to return probabilities

            memory.append((copy.deepcopy(env.board), mcts_prob, env.get_player()))

            
            #TODO: add temperature
            action = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p=mcts_prob)
            # print("action: ", action)
            
            _, reward, done = env.step(action, player=1) # TODO: can be the cause of the bug, why is not player involved here? Compare this...
            # state = env.get_encoded_state()
                    
        # print(state)
        # if env.get_player() == -1:
        if env.get_player() == 1:
            reward *= -1
        return_memory = []
        for state, mcts_prob, player in memory:
            return_memory.append((env.get_encoded_state(state), mcts_prob, reward * player))
        # print(return_memory)
        # print("Game over! player : " , env.get_player(), " Won! on turn: ", env.turn)
        #print("First reward in memory: ", return_memory[0][2])
        return return_memory

            
if __name__ == "__main__":
    # arr1= np.array([1, 2, 3, 4])
    # arr2 = np.array([True, True, False, False])
    # arr1*=arr2
    # print(arr1)
    
    # env=ConnectFour()
    # mcts = MCTSNN()
    # actions = mcts.get_action(env=ConnectFour(), training_return=True)
    # action_prob = actions / np.sum(actions)
    # print(action_prob)
    trainer = Trainer()

    training_iterations = 0
    games = 10
    depth = 5_000
    
    while True:
        print("Training iteration: ", training_iterations)
        try:
            trainer.train(num_games=games, nn_depth=depth)
        except KeyboardInterrupt:
            trainer.save_model_and_optimizer(depth=depth, games=games)
            break
        training_iterations += 1
    
    
    
