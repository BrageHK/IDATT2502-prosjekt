import numpy as np

class Node:
    def __init__(self, env, state, parent=None, action_taken=None):
        if not env.check_state_format(state):
            print("ERROR: In state format Node constructor")
        
        self.children = []
        self.parent = parent
        self.action_taken = action_taken
        self.env = env
        self.reward = 0
        self.visits = 0
        self.state = state
        
        self.c = 4  # Exploration parameter
        
        self.expandable_moves = env.get_legal_moves_bool_array(state)
        
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        node = self
        best_child = self
        while node.is_fully_expanded():
            max_UCB = float('-inf')
            for child in node.children:
                UCB_value = node.get_ucb(child)
                if UCB_value > max_UCB:
                    max_UCB = UCB_value
                    best_child = child
            node = best_child
        return best_child
    
    def get_ucb(self, child): # TODO: rewrite
        q_value = 1 - ((child.reward / child.visits) + 1) / 2
        return q_value + self.c * np.sqrt(np.log(self.visits) / child.visits)
    
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        child_state = self.state.copy()
        child_state, reward, done = self.env.step(child_state, action, 1)
        child_state = self.env.change_perspective(child_state, player=-1)
        
        child = Node(self.env, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.env.check_game_over(self.state, self.action_taken)
        value = self.env.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.env.get_legal_moves_bool_array(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state, reward, done = self.env.step(rollout_state, action, rollout_player)
            value, is_terminal = self.env.check_game_over(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.env.get_opponent_value(value)
                return value    
            
            rollout_player = self.env.get_opponent(rollout_player)
            
    def backpropagate(self, reward):
        self.reward += reward
        self.visits += 1
        
        if self.parent:
            opponent_reward = self.env.get_opponent_value(reward)
            self.parent.backpropagate(opponent_reward)  

        