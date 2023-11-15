import torch.multiprocessing as mp
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
import random

from MCTS.MCTS import MCTS
from Connect_four_env import ConnectFour
from TicTacToe import TicTacToe
from NeuralNet import AlphaPredictorNerualNet
from Node.NodeType import NodeType


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
            mcts_prob, action = mcts.search(neutral_state, training=True) 

            memory.append((neutral_state, mcts_prob, player))

            mcts_prob = np.power(mcts_prob, 1/1.25)
            mcts_prob = mcts_prob / np.sum(mcts_prob)
            action = np.random.choice(env.action_space, p=mcts_prob)
            
            state, reward, done = env.step(state, action=action, player=player)
            if done:
                return_memory = []
                for historical_state, historical_mcts_prob, historical_player in memory:
                    historical_outcome = reward if historical_player == player else env.get_opponent_value(reward)
                    return_memory.append((env.get_encoded_state(historical_state), historical_mcts_prob, historical_outcome))
                return return_memory
            
            player = env.get_opponent(player)
            
            turn += 1        

class Trainer:
    def __init__(self, model, env=ConnectFour(), num_iterations=600):
        self.policy_loss_history = []
        self.value_loss_history = []
        self.game_length_history = []
        self.model = model
        self.mcts = MCTS(env, num_iterations, NODE_TYPE=NodeType.NODE_NN, model=model)
        self.env = env
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        print("Starting training using: ", "cuda" if torch.cuda.is_available() else "cpu")
        print("Cores used for training: ", mp.cpu_count())
        
        
    def save_game_length(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.game_length_history, file)
        
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def save_optimizer(self, filename):
        torch.save(self.optimizer.state_dict(), filename)
        
    def save_loss_history(self, filename):
        with open(filename, "wb") as file:
            pickle.dump((self.policy_loss_history, self.value_loss_history), file)
        
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
            self.model.policy_loss_history, self.model.value_loss_history = self.load_loss_history(filename_loss_values)
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

        print("Starting self play")
        mp.set_start_method('spawn', force=True)
        with mp.Pool(mp.cpu_count()) as pool:
            result_list = pool.starmap(play_game, args_list)
        
        for i in range(len(result_list)):
            memory += result_list[i]
            self.game_length_history.append(len(result_list[i]))
            
        print("Training model")
        self.model.train() # Sets training mode
        
        self.optimize(memory)
        
        return memory
    
    def loss(self, policy_logits, value_logits, policy_target, value_target): #
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_logits, value_target)
        self.value_loss_history.append(value_loss.item())
        self.policy_loss_history.append(policy_loss.item())
        return policy_loss + value_loss
    
    def optimize(self, memory, epoch=4, batch_size=128, training_positions=50_000):
        for i in range(epoch):
            random.shuffle(memory)

            print("Starting epoch: ", i+1)
            for batch_index in range(max(0, len(memory) - training_positions), len(memory), batch_size):
                sample = memory[batch_index:min(len(memory) - 1, batch_index + batch_size)]
                states, policy_targets, value_targets = zip(*sample)
                
                states = np.array(states)
                policy_targets = np.array(policy_targets)
                value_targets = np.array([np.array(item).reshape(-1, 1) for item in value_targets])
            
                states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
                policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
                value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
                
                policy_output, value_output = self.model(states)
                
                loss = self.loss(policy_output, value_output, policy_targets, value_targets.squeeze(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    num_resBlocks = 4
    model = AlphaPredictorNerualNet(num_resBlocks=num_resBlocks, device=device, env=env)
    trainer = Trainer(env = env, model=model, num_iterations=60)

    games = 500 #mp.cpu_count()
    
    folder = "data/"+env.__repr__()+"/"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder created: {folder}")
    
    filename_model = folder+f"model-{num_resBlocks}.pt"
    filename_optimizer = folder+f"optimizer-{num_resBlocks}.pt"
    filename_loss_values = folder+f"loss_values-{num_resBlocks}.pk1"
    filename_game_length = folder+f"game_length-{num_resBlocks}.pk1"
        
    load_all = False
    if load_all:
        trainer.load_data(
            filename_model=filename_model, 
            filename_loss_values=filename_loss_values, 
            filename_optimizer=filename_optimizer, #Funker ikke å laste inn adam fordi idk
            filename_game_length=filename_game_length
            )
        
    def save_all():
        print("\nSaving model, optimizer and loss values")
        trainer.save_model(folder+f"model-{num_resBlocks}.pt")
        trainer.save_loss_history(folder+f"loss_values-{num_resBlocks}.pk1")
        trainer.save_optimizer(folder+f"optimizer-{num_resBlocks}.pt")
        trainer.save_game_length(folder+f"game_length-{num_resBlocks}.pk1")
        print("Saved!")
    
    def save_all_iterations(iteration):
        print("\nSaving model, optimizer, game lengths and loss values")
        trainer.save_model(folder+f"model-{num_resBlocks}-{iteration}.pt")
        trainer.save_loss_history(folder+f"loss_values-{num_resBlocks}-{iteration}.pk1")
        trainer.save_optimizer(folder+f"optimizer-{num_resBlocks}-{iteration}.pt")
        trainer.save_game_length(folder+f"game_length-{num_resBlocks}-{iteration}.pk1")
        print("Saved!")

    training_iterations = 0
    while training_iterations < 3:
        print("Training iteration: ", training_iterations)
        try:
            trainer.train(num_games=games)
        except KeyboardInterrupt:
            save_all()
            break
        save_all()
        training_iterations += 1
        
        print("Loss values:")
        
        # avg loss of last 10 elements
        print("avg loss of last 10 elements")
        print("policy loss: ", sum(trainer.policy_loss_history[-10:])/10)
        print("value loss: ", sum(trainer.value_loss_history[-10:])/10)
        
        if training_iterations % 20 == 0:
            save_all_iterations(training_iterations)
        
        if training_iterations % 30:
            memory = []