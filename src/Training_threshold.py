from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from collections import deque
import numpy as np
import torch
from NeuralNetThreshold import NeuralNetThreshold

from Node.NodeType import NodeType

import pickle

class TrainerThreshold:
    def __init__(self, env=ConnectFour(), num_iterations=20_000, model=NeuralNetThreshold()):
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
        print("Starting self play")
        for i in range(num_games):
            print("Playing game: ", i)
            memory.extend(self.play_game())
            
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
            
    def play_game(self):
        memory = []
        player = 1
        done = False
        state = self.env.get_initial_state()
        
        while not done:
            state = self.env.change_perspective(state, player)
            action = self.mcts.search(state)

            memory.append((state, player))
            
            state, reward, done = self.env.step(state, action=action, player=1) # TODO: can be the cause of the bug, why is not player involved here? Compare this..
            player = self.env.get_opponent(player)
            
        return_memory = []
        for historical_state, historical_player in memory:
            if historical_player == player: # player because the while loop has switch the player
                reward = self.env.get_opponent_value(reward)
            return_memory.append((self.env.get_encoded_state(historical_state), self.reward_tuple(reward)))
        return return_memory
    
    def save_games(self, memory, filename):
        with open(filename, "wb") as file:
            pickle.dump(memory, file)
            
    def load_games(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


            
if __name__ == "__main__":
    trainer = TrainerThreshold(env = ConnectFour())

    training_iterations = 0
    games = 25
    memory = deque(maxlen=500_000)
    filename = "data/threshold/model.pt"
    filename_games = "data/threshold/games.pk1"
    filename_loss_values = "data/threshold/loss_values.pk1"

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
    