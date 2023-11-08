from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from collections import deque
import numpy as np
import torch
from NeuralNet import AlphaPredictorNerualNet

from Node.NodeType import NodeType

import pickle

class Trainer:
    def __init__(self, env=ConnectFour(), num_iterations=500, model=AlphaPredictorNerualNet(4)):
        self.model = model
        self.mcts = MCTS(env, num_iterations, NODE_TYPE=NodeType.NODE_NN, model=model)
        self.env = env
    
    def save_model_and_optimizer(self, filename):
        torch.save(self.model.state_dict(), filename)
        #torch.save(self.model.optimizer.state_dict(), f"optimizer-{depth}-depth-and-{games}-games.pt")
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, num_games, memory):
        self.model.eval()
        
        # Plays num_games against itself and stores the data in memory
        for i in range(num_games):
            print("Playing game: ", i)
            memory += self.play_game()
            
        # Trains the model on the data in memory
        print("Training model")
        self.model.train() # Sets training mode
        
        self.model.optimize(self.model, memory)
        
        return memory
            
    def play_game(self):
        memory = []
        player = 1
        done = False
        state = self.env.get_initial_state()
        
        while not done:
            state = self.env.change_perspective(state, player)
            mcts_prob, action = self.mcts.search(state, training=True) 

            memory.append((state, mcts_prob, player))

            #TODO: add temperature
            action = np.random.choice(self.env.action_space, p=mcts_prob)
            
            state, reward, done = self.env.step(state, action=action, player=1) # TODO: can be the cause of the bug, why is not player involved here? Compare this..
            player = self.env.get_opponent(player)
            
        return_memory = []
        for historical_state, historical_mcts_prob, historical_player in memory:
            if historical_player == player: # player because the while loop has switch the player
                reward = self.env.get_opponent_value(reward)
            return_memory.append((self.env.get_encoded_state(historical_state), historical_mcts_prob, reward))
        return return_memory
    
    def save_games(self, memory, filename):
        with open(filename, "wb") as file:
            pickle.dump(memory, file)
            
    def load_games(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


            
if __name__ == "__main__":
    trainer = Trainer(env = ConnectFour())

    training_iterations = 0
    games = 20
    memory = []
    filename = "model.pt"
    filename_games = "games.pk1"

    try:
        trainer.load_model(filename)
    except FileNotFoundError:
        print("No model found with name: ", filename)
    
    try:
        memory = trainer.load_games(filename_games)
    except FileNotFoundError:
        print("No memory found")
    
    while True:
        print("Training iteration: ", training_iterations)
        try:
            memory+=trainer.train(num_games=games, memory=memory)
        except KeyboardInterrupt:
            print("\nSaving model and games")
            trainer.save_model_and_optimizer(filename)
            trainer.save_games(memory, filename_games)
            break
        
        if training_iterations % 20 == 0:
            print("\nSaving model and games")
            trainer.save_model_and_optimizer(filename)
            trainer.save_games(memory, filename_games)
        
        training_iterations += 1
    