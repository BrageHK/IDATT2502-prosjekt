from MCTS_NN import MCTSNN
from Connect_four_env import ConnectFour
from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from NeuralNet import AlphaPredictorNerualNet

class Trainer:
    def __init__(self, model=AlphaPredictorNerualNet(4)):
        self.model = model
        self.mcts = MCTSNN(model)

    def train(self, num_games, nn_depth, epochs):
        memory = []
        self.model.eval()
        for _ in range(num_games):
            memory += self.play_game(nn_depth)
            
        
        
            
        self.model.train() # Sets training mode
        for epoch in range(epochs):
            
            pass
        torch.save(self.model.state_dict(), "model.pt")
            
    def play_game(self, nn_depth):
        env = ConnectFour()
        memory = []
        invert = False # Player 1 first makes a move
        done = False
        reward = 0
        
        while not done:
            mcts_prob, action = self.mcts.get_action(env=env, n_simulations=nn_depth, invert=invert, training_return=True)
            memory.append((env.board, mcts_prob, env.get_player()))
            
            action = np.random.choice(env.get_legal_moves(), p=mcts_prob)
            print("action: ", action)
            _, reward, done = env.step(action)
            invert = not invert
        
        reward = -reward 
        return_memory = []
        for state, mcts_prob, player in memory:
            return_memory.append((env.get_encoded_state(state), mcts_prob, reward * player))
        print(return_memory)
        print("Game over! player : " , env.get_player(), " Won!")
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
    
    trainer.train(1, 7, 1)
    
    
