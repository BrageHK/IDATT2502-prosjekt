from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from collections import deque
import numpy as np
import torch

from NeuralNetThreshold import NeuralNetThreshold
from board_printer import BoardPrinter
from Node.NodeType import NodeType

import pickle
import os

import torch.multiprocessing as mp

import time

def play_game(env, mcts, match_id):
        print(f'Starting match {match_id}')

        memory = []
        player = 1
        done = False
        state = env.get_initial_state()
        
        turn = 0
        
        while not done:
            neutral_state = env.change_perspective(state, player)
            action = mcts.search(neutral_state)

            memory.append((neutral_state, player))
            
            state, reward, done = env.step(state, action=action, player=player) # TODO: can be the cause of the bug, why is not player involved here? Compare this..
            player = env.get_opponent(player)
            
            turn += 1
            if match_id == 1:
                print("Turn: ", turn)
            
        player = env.get_opponent(player)
        return_memory = []
        for historical_state, historical_player in memory:
            historical_outcome = reward if historical_player == player else env.get_opponent_value(reward)
            return_memory.append((env.get_encoded_state(historical_state), historical_outcome))
        return return_memory

class TrainerThreshold:
    def __init__(self, env=ConnectFour(), num_iterations=10_000, model=NeuralNetThreshold()):
        self.model = model
        self.mcts = MCTS(env, num_iterations, NODE_TYPE=NodeType.NODE_THRESHOLD, model=model)
        self.env = env
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
         
    def load_model(self, path):
        self.model.load_model(path)

    # Memory should be a deque with 500_000 length
    def train(self, num_games, memory):
        self.model.eval() # Sets evaluation mode
        
        # Plays num_games against itself and stores the data in memory
        mp.set_start_method('spawn', force=True)

        match_id = 1
        args_list = []
        for _ in range(num_games):
            args_list.append((self.env, self.mcts, match_id))
            match_id += 1
        #mp.cpu_count()
        # 1 core for debug
        
        print("Starting self play")
        starting_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            result_list = pool.starmap(play_game, args_list)
        print("games done i n: ", time.time()-starting_time, " seconds")
        for i in range(len(result_list)):
            memory.extend(result_list[i])
            
        BoardPrinter.memory_debugger(result_list[0])
            
        # Trains the model on the data in memory
        print("Training model")
        self.model.train() # Sets training mode
        
        self.model.optimize(self.model, memory)
        
        return memory
    
    def reward_tuple(self, reward):
        if reward == 1:
            return (1, 0, 0)
        elif reward == 0:
            return (0, 1, 0)
        else:
            return (0, 0, 1)
            
    
    
    def save_games(self, memory, filename):
        with open(filename, "wb") as file:
            pickle.dump(memory, file)
            
    def load_games(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

if __name__ == "__main__":
    trainer = TrainerThreshold(env = ConnectFour())

    training_iterations = 0
    games = mp.cpu_count()
    memory = deque(maxlen=500_000)
    folder = "data/threshold/"
    filename = folder+"model.pt"
    filename_games = folder+"games.pk1"
    filename_loss_values = folder+"loss_values.pk1"
    
    if not os.path.exists(folder):
        # Create the folder
        os.makedirs(folder)
        print(f"Folder created: {folder}")

    try:
        trainer.load_model(filename)
    except FileNotFoundError:
        print("No model found with name: ", filename)
    
    try:
        memory = trainer.load_games(filename_games)
    except FileNotFoundError:
        print("No memory found from file: ", filename_games)
    
    while True:
        print("Training iteration: ", training_iterations)
        try:
            memory.extend(trainer.train(num_games=games, memory=memory))
        except KeyboardInterrupt:
            print("\nSaving model")
            trainer.save_model(filename)
            print("Saving games")
            trainer.save_games(memory, filename_games)
            print("Saving loss values")
            trainer.model.save_loss_values_to_file(filename_loss_values)
            break
        
        if training_iterations % 10 == 0:
            print("\nSaving model, games and loss values")
            trainer.save_model(filename)
            trainer.save_games(memory, filename_games)
            trainer.model.save_loss_values_to_file(filename_loss_values)
        
        training_iterations += 1
    