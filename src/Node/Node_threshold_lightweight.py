import numpy as np
import torch
from  NeuralNetThreshold import NeuralNetThreshold

class NodeThresholdLightweight:
    def __init__(self, env, state, model, parent=None, action_taken=None):
        if not env.check_state_format(state):
            print("ERROR: In state format Node constructor")
        
        self.children = []
        self.parent = parent
        self.action_taken = action_taken
        self.env = env
        self.reward = 0
        self.visits = 0
        self.state = state
        self.model = model
        
        self.c = 4  # Exploration parameter
        self.threshold = 0.9
        
        self.expandable_moves = env.get_legal_moves_bool_array(state)
        
    
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        node = self
        best_child = self
        while node.is_fully_expanded():
            max_UCB = float('-inf')
            for child in node.children:
                UCB_value = node.calculate_UCB(child)
                if UCB_value > max_UCB:
                    max_UCB = UCB_value
                    best_child = child
            node = best_child
        return best_child
    
    def calculate_UCB(self, child): # TODO: rewrite
        if child.visits == 0:
            return float('inf')
    
        # Add 1 and divide by 2 to always have q_value between 0 and 1. If this is not done the q_value can be between -1 and 1
        # 1 - expression becuase the view is from the parent and not the child
        
        q_value = 1 - ((child.reward / child.visits) + 1) / 2 
        return q_value + np.sqrt(self.c * np.log(self.visits) / child.visits)
        
    def expand(self):
        if self.visits > 0:
            legal_moves = self.env.get_legal_moves(self.state)
            for action in legal_moves:
                self.expandable_moves[action] = 0
                child_state = self.state.copy()
                child_state, reward, done = self.env.step(child_state, action, 1)
                child_state = child_state = self.env.change_perspective(child_state, player=-1)
                
                child = NodeThresholdLightweight(env=self.env, state=child_state, parent=self, action_taken=action, model=self.model)
                self.children.append(child)
    
    def random_action(self, env, state):
        return np.random.choice(env.get_legal_moves(state))
    
    def get_reward_from_value(self, value):
        if value == 0:
            return 1
        elif value == 1:
            return 0
        elif value == 2:
            return -1
        else:
            raise Exception("ERROR: Invalid value in get_reward_from_value")
    
    def simulate(self):
        reward, done = self.env.check_game_over(self.state, self.action_taken)
        reward = self.env.get_opponent_value(reward)
        
        if done:
            return reward
        
        rollout_state = self.state.copy()
        rollout_player = 1
        
        while not done:
            values = self.model.forward(torch.tensor(rollout_state, device=self.model.device, dtype=torch.float32,).flatten()) # ai predicts win, draw or loss in this position
            for i in range(len(values)): # if value is over treshold, no point in continuing simulating
                if values[i].item() > self.threshold:
                    reward = self.get_reward_from_value(i)
                    break
                    
            rollout_action = self.random_action(self.env, rollout_state)
            rollout_state, reward, done = self.env.step(rollout_state, rollout_action, rollout_player)
            rollout_player = self.env.get_opponent(rollout_player)
            
        if self.env.get_opponent(rollout_player) == -1:
            reward = self.env.get_opponent_value(reward)
        return reward, rollout_state 
            
    def backpropagate(self, reward):
        self.reward += reward
        self.visits += 1
        
        if self.parent:
            opponent_reward = self.env.get_opponent_value(reward)
            self.parent.backpropagate(opponent_reward)

        