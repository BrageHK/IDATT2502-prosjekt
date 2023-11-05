import numpy as np
from Connect_four_env import ConnectFour


class NodeDouble():
    
    def __init__(self, parent=None, env=ConnectFour(), action=None):
        self.children = []
        self.parent = parent
        self.action = action
        self.env = env
        self.reward1 = 0 # Player 1 reward
        self.reward2 = 0 # Player 2 reward
        self.visits = 0
        
        self.c = 4 # Exploration parameter
        
    def calculate_UCB(self, child):
        if child.visits == 0:
            return float('inf')
        
        return self.env.get_player() * np.abs(child.reward1 - child.reward2) / child.visits + np.sqrt(self.c * np.log(self.visits) / child.visits)

    # def select(self):
    #     node = self
    #     while bool(node.children): 
    #         UCB_values = []
    #         for child in node.children:
    #             UCB_values.append(node.calculate_UCB(child))
    #         node = node.children[np.argmax(UCB_values)]
    
    #     return node
    
    def select(self):
        node = self
        best_child = self
        while bool(node.children):
            max_UCB = float('-inf')
            for child in node.children:
                UCB_value = node.calculate_UCB(child)
                if UCB_value > max_UCB:
                    max_UCB = UCB_value
                    best_child = child
            node = best_child
        return best_child

    # Adds a child node for untried action and returns it
    def expand(self):
        if self.visits > 0:
            legal_moves = self.env.get_legal_moves()
            for action in legal_moves:
                next_env = self.env.deepcopy()
                next_env.step(action)
                child = NodeDouble(parent=self, env=next_env, action=action)
                
                self.children.append(child)
            
    def random_act(self, env):
        return np.random.choice(env.get_legal_moves())

    # Simulate random actions until a terminal state is reached
    def simulate(self):
        simulated_game = self.env.deepcopy()
        current_player = simulated_game.get_last_player() # last player because step is called in expand
        reward, done = simulated_game.check_game_over(current_player)
            
        while not done:
            action = self.random_act(simulated_game)
            reward, done = simulated_game.step(action)

        winning_player = simulated_game.get_last_player() # last player becuase the last player won
        
        # print("Board:\n")
        # print(simulated_game.board)
        # if simulated_game.is_inverted:
        #     print("\n",-simulated_game.board)
            
        
        return reward * winning_player, simulated_game.board # Return 1 if the player won, -1 if the player lost, and 0 if it was a draw.
    
    def backpropagate(self, result):
        self.visits += 1
        if result == 1:
            self.reward1 += 1
        elif result == -1:
            self.reward2 += 1
        
        if self.parent:
            self.parent.backpropagate(result)