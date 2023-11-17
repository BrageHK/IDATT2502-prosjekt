from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from TicTacToe import TicTacToe
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os
import torch.multiprocessing as mp
import time
import random

from NeuralNetThreshold import NeuralNetThreshold
from Node.NodeType import NodeType

def reward_tuple(reward):
        if reward == 1:
            return (1, 0, 0) # win
        elif reward == 0:
            return (0, 1, 0) # draw 
        else:
            return (0, 0, 1) # loss

@torch.no_grad()
def play_game(env, mcts, match_id):
        print(f'Starting match {match_id}')

        memory = []
        player = 1
        done = False
        state = env.get_initial_state()
        turn = 0

        while True:
            neutral_state = env.change_perspective(state, player)
            action = mcts.search(neutral_state)
            
            memory.append((neutral_state, player))
            
            state, reward, done = env.step(state, action=action, player=player)
            
            if done:
                return_memory = []
                for historical_state, historical_player in memory:
                    historical_outcome = reward if historical_player == player else env.get_opponent_value(reward)
                    historical_outcome = reward_tuple(historical_outcome)
                    return_memory.append((env.get_encoded_state(historical_state), historical_outcome))
                return return_memory
            
            turn += 1
            player = env.get_opponent(player)

class TrainerThreshold:
    def __init__(self, model, env=ConnectFour(), num_iterations=5_000):
        self.model = model
        self.mcts = MCTS(env, num_iterations, NODE_TYPE=NodeType.NODE_THRESHOLD, model=model)
        self.env = env
        self.value_loss_history = []
        self.game_length_history = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        print("Starting training using: ", "cuda" if torch.cuda.is_available() else "cpu")
        print("Cores used for training: ", mp.cpu_count())
    
    def save_game_length(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.game_length_history, file)
        
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def save_loss_history(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.value_loss_history, file)
    
    def save_optimizer(self, filename):
        torch.save(self.optimizer.state_dict(), filename)
         
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def load_loss_history(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
    
    def load_optimizer(self, filename):
        self.optimizer.load_state_dict(torch.load(filename))
    
    def load_game_length(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
        
    def load_data(self, filename_model, filename_loss_values, filename_optimizer, filename_game_length):
        try:
            self.load_model(filename_model)
        except FileNotFoundError:
            print("No model found with name: ", filename_model)
        try:
            self.optimizer = self.load_optimizer(filename_optimizer)
        except FileNotFoundError:
            print("No loss values found from file: ", filename_optimizer)
        try:
            self.model.value_loss_history = self.load_loss_history(filename_loss_values)
        except FileNotFoundError:
            print("No loss values found from file: ", filename_loss_values)
        try:
            self.game_length_history = self.load_game_length(filename_game_length)
        except FileNotFoundError:
            print("No game length values found from file: ", filename_game_length)

    def train(self, num_games, memory=[]):
        self.model.eval()
        match_id = 1
        args_list = []
        for _ in range(num_games):
            args_list.append((self.env, self.mcts, match_id))
            match_id += 1
        
        mp.set_start_method('spawn', force=True)
        
        with mp.Pool(mp.cpu_count()) as pool:
            result_list = pool.starmap(play_game, args_list)
        for i in range(len(result_list)):
            memory.extend(result_list[i])
        
        print("Training model")
        self.model.train() # Sets training mode
        self.optimize(memory=memory)
        
        return memory
    
    def loss(self, value_logits, value_target): #
        value_loss = F.mse_loss(value_logits, value_target)
        self.value_loss_history.append(value_loss.item())
        return value_loss
    
    def optimize(self, memory, epoch=4, batch_size=128):
        for i in range(epoch):
            random.shuffle(memory)
            
            print("Starting epoch: ", i+1)
            for batch_index in range(0, len(memory), batch_size):
                sample = memory[batch_index:min(len(memory) - 1, batch_index + batch_size)]
                states, value_targets = zip(*sample)
                
                states = np.array(states)
                value_targets = np.array([np.array(item).reshape(-1, 1) for item in value_targets])
            
                states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
                value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
                
                value_output = self.model(states)
                
                loss = self.loss(value_output, value_targets.squeeze(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def save_games(self, memory, filename):
        with open(filename, "wb") as file:
            pickle.dump(memory, file)
            
    def load_games(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder created: {folder}")
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    model = NeuralNetThreshold(env=env, device=device)
    trainer = TrainerThreshold(env = env, model=model, num_iterations=10_000)

    games = mp.cpu_count()*8
    
    folder = "data/threshold"+env.__repr__()+"/"
    
    folders = ["","model", "loss", "optimizer", "game_length"]
    
    for path in folders:
        create_folder(folder+path)
    
    filename_model = folder+f"model.pt"
    filename_optimizer = folder+f"optimizer.pt"
    filename_loss_values = folder+f"loss_values.pk1"
    filename_game_length = folder+f"game_length.pk1"
        
    load_all = False
    if load_all:
        trainer.load_data(
            filename_model=filename_model, 
            filename_loss_values=filename_loss_values, 
            filename_optimizer=filename_optimizer,
            filename_game_length=filename_game_length
            )
        
    def save_all():
        print("\nSaving model, optimizer and loss values")
        trainer.save_model(folder+f"model.pt")
        trainer.save_loss_history(folder+f"loss_values.pk1")
        trainer.save_optimizer(folder+f"optimizer.pt")
        trainer.save_game_length(folder+f"game_length.pk1")
        print("Saved!")
    
    def save_all_iterations(iteration):
        print("\nSaving model, optimizer, game lengths and loss values")
        trainer.save_model(folder+f"model/model-{iteration}.pt")
        trainer.save_loss_history(folder+f"loss/loss_values-{iteration}.pk1")
        trainer.save_optimizer(folder+f"optimizer/optimizer-{iteration}.pt")
        trainer.save_game_length(folder+f"game_length/game_length-{iteration}.pk1")
        print("Saved!")

    training_iterations = 0
    while True:
        print("Training iteration: ", training_iterations)
        try:
            trainer.train(num_games=games)
        except KeyboardInterrupt:
            save_all()
            break
        save_all()
        training_iterations += 1
        
        if training_iterations % 5 == 0:
            save_all_iterations(training_iterations)