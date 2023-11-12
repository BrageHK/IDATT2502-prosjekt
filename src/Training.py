import torch.multiprocessing as mp

from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from collections import deque

from NeuralNet import AlphaPredictorNerualNet
from Node.NodeType import NodeType

import pickle
import numpy as np
import torch
import os

def play_game(env, mcts, match_id):
        print(f'Starting match {match_id}')

        memory = []
        player = 1
        done = False
        state = env.get_initial_state()
        turn = 0
        
        while not done:
            state = env.change_perspective(state, player)
            mcts_prob, action = mcts.search(state, training=True) 

            memory.append((state, mcts_prob, player))

            if turn < 16: # Higher exploration in the first 10 moves
                mcts_prob = np.power(mcts_prob, 1/1.5)
                mcts_prob = mcts_prob / np.sum(mcts_prob)
            action = np.random.choice(env.action_space, p=mcts_prob)
            
            state, reward, done = env.step(state, action=action, player=1)
            player = env.get_opponent(player)
            
            turn += 1
        
        player = env.get_opponent(player)
        return_memory = []
        for historical_state, historical_mcts_prob, historical_player in memory:
            historical_outcome = reward if historical_player == player else env.get_opponent_value(reward)
            return_memory.append((env.get_encoded_state(historical_state), historical_mcts_prob, historical_outcome))
        return return_memory

class Trainer:
    def __init__(self, env=ConnectFour(), num_iterations=1_000, model=AlphaPredictorNerualNet(4)): # TODO: change iterations to 1000
        
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

    def train(self, num_games, memory):
        print("self.model.eval()")
        self.model.eval()
        
        # Plays num_games against itself and stores the data in memory
        print("Starting self play")
        
        # for i in range(num_games):
        #     print("Started game: ", i)
        #     memory.extend(self.play_game())@
        
        mp.set_start_method('spawn', force=True)

        match_id = 1
        args_list = []
        for _ in range(num_games):
            args_list.append((self.env, self.mcts, match_id))
            match_id += 1
        #mp.cpu_count()
        # 1 core for debug
        with mp.Pool(mp.cpu_count()) as pool:
            result_list = pool.starmap(play_game, args_list)
        
        for i in range(len(result_list)):
            memory.extend(result_list[i])
            
        # Trains the model on the data in memory
        print("Training model")
        self.model.train() # Sets training mode
        
        self.model.optimize(self.model, memory)
        
        return memory
    
    def save_games(self, memory, filename):
        with open(filename, "wb") as file:
            pickle.dump(memory, file)
            
    def load_games(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
        
    def load_loss_history(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


def load_data(filename, filename_games, filename_loss_values, trainer):
    memory = deque(maxlen=50_000)
    try:
        trainer.load_model(filename)
    except FileNotFoundError:
        print("No model found with name: ", filename)
    try:
        memory = trainer.load_games(filename_games)
    except FileNotFoundError:
        print("No memory found from file: ", filename_games)
        exit()
    try:
        trainer.model.policy_loss_history, trainer.model.value_loss_history = trainer.load_loss_history(filename_loss_values)
    except FileNotFoundError:
        print("No loss values found from file: ", filename_loss_values)
    return memory

if __name__ == "__main__":
    
    print("Starting training file")
    
    trainer = Trainer(env = ConnectFour())

    training_iterations = 0
    games = mp.cpu_count()
    memory = deque(maxlen=500_000)
    
    folder = "data/test/"
    
    if not os.path.exists(folder):
        # Create the folder
        os.makedirs(folder)
        print(f"Folder created: {folder}")
    
    filename = folder+"model.pt"
    filename_games = folder+"games.pk1"
    filename_loss_values = folder+"loss_values.pk1"
        
    load_all = False
    if load_all:
        memory = load_data(filename, filename_games, filename_loss_values, trainer)
        
    def save_all():
        print("\nSaving model, games and loss values")
        trainer.save_model(filename)
        trainer.save_games(memory, filename_games)
        trainer.model.save_loss_values_to_file(filename_loss_values)
        print("Saved!")
        pass
    
    while training_iterations < 10:
        print("Training iteration: ", training_iterations)
        try:
            memory.extend(trainer.train(num_games=games, memory=memory))
        except KeyboardInterrupt:
            save_all()
            break
        
        save_all()
        
        training_iterations += 1