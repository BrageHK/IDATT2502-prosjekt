from MCTS_NN import MCTSNN
from Connect_four_env import ConnectFour
from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from NeuralNet import AlphaPredictorNerualNet

class Trainer:
    def __init__(self):
        pass

    def train(self, num_games, nn_depth, epochs):
        model = AlphaPredictorNerualNet(4)
        
        memory = []
        model.eval()
        for _ in range(num_games):
            memory += self.play_game(nn_depth, model)
            
        print(memory)
            
        model.train() # Sets training mode
        for epoch in range(epochs):
            
            pass
        torch.save(model.state_dict(), "model.pt")
            
    def play_game(self, nn_depth, model):
        env = ConnectFour()
        mcts = MCTSNN(model)
            
        memory = []
        # Player 1 first makes a move
        invert = False
        done = False
        reward = 0
        
        state = env.get_encoded_state()
        
        while True:
            mcts_prob, action = mcts.get_action(env=env, n_simulations=nn_depth, invert=invert, training_return=True)  # assuming your MCTS has an option to return probabilities
            action = np.random.choice(env.get_legal_moves(), p=mcts_prob)
            print("action: ", action)
            memory.append((state, mcts_prob, env.get_player()))
            
            state, reward, done = env.step(action)
            state = env.get_encoded_state()
            invert = not invert
            
            if done:
                print("Game over! player : " , env.get_player(), " Won!")
                reward = -reward * env.get_player()
                return_memory = []
                for state, mcts_prob, player in memory:
                    return_memory.append((state, mcts_prob, reward * player))
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
    
    trainer.train(1, 1000, 1)
    
    
