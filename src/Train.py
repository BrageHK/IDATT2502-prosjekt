from MCTS_NN import MCTSNN
from Connect_four_env import ConnectFour
from collections import deque

class Trainer:
    def __init__(self, NN):
        self.NN = NN

    def train(self, num_games, nn_depth):
        
        env=ConnectFour()
        mcts = MCTSNN()
        for _ in range(num_games):
            
            invert = False
            game_states = []
            actions = []
            reward = 0
            while not done:
                # Player 1
                action = mcts.get_action(env=env, n_simulations=nn_depth, invert=invert)
                state, reward, done = env.step(action)
                game_states.append(state)
                actions.append(action)
                invert = not invert
            
            reward = reward * env.get_player() # 1 for player 1 win, 2 for player 2 win, 0 for draw
            rewards = [1 if i%2==0 else -1 for i in range(len(game_states))]
            
                
            for state, reward, done in range(game_states):
                games = tuple(game_states, actions, reward)
            
            games = []
            
            # Train NN
            