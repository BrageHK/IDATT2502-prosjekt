import numpy as np

class NodeDouble:
    def __init__(self, env, state, parent=None, action_taken=None):
        if not env.check_state_format(state):
            print("ERROR: In state format Node constructor")
        
        self.children = []
        self.parent = parent
        self.action_taken = action_taken
        self.env = env
        self.reward1 = 0 # Player 1 reward
        self.reward2 = 0 # Player 2 reward
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
        if child.visits == 0:
            return float('inf')

        return -np.abs(child.reward1 - child.reward2) / child.visits + np.sqrt(self.c * np.log(self.visits) / child.visits)
        
    def expand(self):
        if self.visits > 0:
            legal_moves = self.env.get_legal_moves(self.state)
            for action in legal_moves:
                self.expandable_moves[action] = 0
                child_state = self.state.copy()
                child_state, reward, done = self.env.step(child_state, action, 1)
                child_state = child_state = self.env.change_perspective(child_state, player=-1)
                
                child = NodeDouble(self.env, child_state, self, action)
                self.children.append(child)
    
    def random_action(self, env, state):
        return np.random.choice(env.get_legal_moves(state))
    
    def simulate(self):
        reward, done = self.env.check_game_over(self.state, self.action_taken)
        reward = self.env.get_opponent_value(reward)
        
        if done:
            return reward
        
        rollout_state = self.state.copy()
        rollout_player = 1
        
        while not done:
            rollout_action = self.random_action(self.env, rollout_state)
            rollout_state, reward, done = self.env.step(rollout_state, rollout_action, rollout_player)
            rollout_player = self.env.get_opponent(rollout_player)
            
        if self.env.get_opponent(rollout_player) == -1:
            reward = self.env.get_opponent_value(reward)
        return reward, rollout_state 
            
    def backpropagate(self, result):
        self.visits += 1
        if result == 1:
            self.reward1 += 1
        elif result == -1:
            self.reward2 += 1
        
        if self.parent:
            # TODO: might get oppoenent value for result?
            self.parent.backpropagate(result)

        