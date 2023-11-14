import torch.multiprocessing as mp

from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from collections import deque

from NeuralNet import AlphaPredictorNerualNet
from Node.NodeType import NodeType
from board_printer import BoardPrinter

import pickle
import numpy as np
import torch
import os

@torch.no_grad()
def play_game(env, mcts, match_id):
        print(f'Starting match {match_id}')

        memory = []
        player = 1
        state = env.get_initial_state()
        turn = 0
        
        while True:
            neutral_state = env.change_perspective(state, player)
            mcts_prob, action = mcts.search(neutral_state, training=True) 

            memory.append((neutral_state, mcts_prob, player))

            if turn < 16: # Higher exploration in the first 10 moves
                mcts_prob = np.power(mcts_prob, 1/1.5)
                mcts_prob = mcts_prob / np.sum(mcts_prob)
            action = np.random.choice(env.action_space, p=mcts_prob)
            
            state, reward, done = env.step(state, action=action, player=player)
            player = env.get_opponent(player)
            
            turn += 1
            if match_id == 1:
                print(f"Turn: {turn}")
            
            if done:
                player = env.get_opponent(player)
                return_memory = []
                for historical_state, historical_mcts_prob, historical_player in memory:
                    historical_outcome = reward if historical_player == player else env.get_opponent_value(reward)
                    return_memory.append(
                (env.get_encoded_state(historical_state), historical_mcts_prob, historical_outcome))
                return return_memory
            player = env.get_opponent(player)

class Trainer:
    def __init__(self, env=ConnectFour(), num_iterations=600, model=AlphaPredictorNerualNet(9)): # TODO: change iterations to 1000
        
        self.model = model
        self.mcts = MCTS(env, num_iterations, NODE_TYPE=NodeType.NODE_NN, model=model)
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Starting training using: ", "cuda" if torch.cuda.is_available() else "cpu")
        print("Cores used for training: ", mp.cpu_count())
        
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, num_games):
        self.model.eval()
        print("Starting self play")
        memory = []
        mp.set_start_method('spawn', force=True)
        match_id = 1
        args_list = []
        for _ in range(num_games):
            args_list.append((self.env, self.mcts, match_id))
            match_id += 1

        with mp.Pool(mp.cpu_count()) as pool:
            result_list = pool.starmap(play_game, args_list)
        
        for i in range(len(result_list)):
            memory += result_list[i]
            
        #BoardPrinter.memory_debugger(result_list[0])
            
        # Trains the model on the data in memory
        print("Training model")
        self.model.train() # Sets training mode
        
        self.model.optimize(self.model, memory)
        
        return memory
        
    def load_loss_history(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


def load_data(filename, filename_loss_values, trainer):
    try:
        trainer.load_model(filename)
    except FileNotFoundError:
        print("No model found with name: ", filename)
    try:
        trainer.model.policy_loss_history, trainer.model.value_loss_history = trainer.load_loss_history(filename_loss_values)
    except FileNotFoundError:
        print("No loss values found from file: ", filename_loss_values)

if __name__ == "__main__":
    
    print("Starting training file")
    
    trainer = Trainer(env = ConnectFour())

    training_iterations = 0
    games = mp.cpu_count()
    
    folder = "data/test2/"
    
    if not os.path.exists(folder):
        # Create the folder
        os.makedirs(folder)
        print(f"Folder created: {folder}")
    
    filename = folder+"model.pt"
    filename_loss_values = folder+"loss_values.pk1"
        
    load_all = True
    if load_all:
        load_data(filename, filename_loss_values, trainer)
        
    def save_all():
        print("\nSaving model, games and loss values")
        trainer.save_model(filename)
        trainer.model.save_loss_values_to_file(filename_loss_values)
        print("Saved!")
        pass
    
    while True:
        print("Training iteration: ", training_iterations)
        try:
            trainer.train(num_games=games)
        except KeyboardInterrupt:
            save_all()
            break
        
        save_all()
        
        training_iterations += 1