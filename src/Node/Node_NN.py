import numpy as np

class NodeNN:
    def __init__(self, env, state, parent=None, action_taken=None, priority=0):
        if not env.check_state_format(state):
            print("ERROR: In state format Node constructor")
        
        self.children = []
        self.parent = parent
        self.action_taken = action_taken
        self.env = env
        self.reward = 0
        self.visits = 0
        self.state = state
        self.priority = priority
        
        self.c = 4  # Exploration parameter
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def calculate_PUCT(self, child):
        # Quality-value
        q_value = 0
        if child.visits != 0:
            q_value = 1 - ((child.reward / child.visits) + 1) / 2
        
        # UCB
        ucb = self.c * child.priority * np.sqrt(np.log(self.visits)) / (1 + child.visits) # TODO: In case of errors try with log inside the square root
        return q_value + ucb
    
    def select(self):
        node = self
        best_child = None
        while node.is_fully_expanded():
            max_UCB = float('-inf')
            for child in node.children:
                UCB_value = node.calculate_PUCT(child)
                if UCB_value > max_UCB:
                    max_UCB = UCB_value
                    best_child = child
            node = best_child
        return node
        
    def expand(self, policy):
        if self.visits > 0:
            for action, probability in enumerate(policy):
                if probability > 0: 
                    child_state = self.state.copy()
                    child_state, reward, done = self.env.step(child_state, action, 1)
                    child_state = child_state = self.env.change_perspective(child_state, player=-1)
                    
                    child = NodeNN(self.env, child_state, self, action, priority=probability)
                    self.children.append(child)
            
    def backpropagate(self, reward):
        self.reward += reward
        self.visits += 1
        
        if self.parent:
            opponent_reward = self.env.get_opponent_value(reward)
            self.parent.backpropagate(opponent_reward)

        