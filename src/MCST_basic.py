import numpy as np
import copy
from ConnectFourEnv import ConnectFour

class Node():
    def __init__(self, parent=None, env=ConnectFour(), action=None):
        self.children = []
        self.parent = parent
        self.action = action
        self.env = env
        self.reward1 = 0
        self.reward2 = 0
        self.visits = 0
        
        self.c = 4 # Exploration parameter
        
    # Upper Confidence Bound fra side 10
    # Verdier bør være mellom 0 og 1 når man vinner eller taper.
    def calculate_UCB(self, child):
        #print("Calculate UCB: ")

        #print("child visits: ", child.visits)
        #print("self visits: ", self.visits)
        
        #print("Reward1: ", child.reward1)
        #print("Reward2: ", child.reward2)

        #print()
        if child.visits == 0:
            return float('inf')
        
        return self.env.get_player() * np.abs(child.reward1 - child.reward2) / child.visits + np.sqrt(self.c * np.log(self.visits) / child.visits)

    def select(self):
        if not bool(self.children):
            return self
        UCB_values = []
        for child in self.children:
            UCB_values.append(self.calculate_UCB(child))
        return self.children[np.argmax(UCB_values)]

    # Adds a child node for untried action and returns it
    def expand(self):
        if self.visits > 0: ## UNDO??
            print("Dette skal vel ikke skje?")
            pass
        legal_moves = self.env.get_legal_moves()
        #print("legal_moves: ", legal_moves)
        for action in legal_moves:
            next_env = copy.deepcopy(self.env)
            board, reward, done = next_env.step(action)
            child = Node(parent=self, env=next_env, action=action)
            self.children.append(child)
            
    def random_act(self, env):
        return np.random.choice(env.get_legal_moves())

    # Simulate random actions until a terminal state is reached
    def simulate(self):
        
        simulated_game = copy.deepcopy(self.env)
        start_simulation = copy.deepcopy(simulated_game)
        current_player = simulated_game.get_player() # -1 or 1
        done = False
        reward, done = simulated_game.check_game_over(current_player)
        
        while not done:
            action = self.random_act(simulated_game)
            board, reward, done = simulated_game.step(action)

        winning_player = simulated_game.get_player()
        return reward * winning_player, simulated_game.board # Return 1 if the player won, -1 if the player lost, and 0 if it was a draw.
    
    def backpropagate(self, reward1, reward2):
        self.visits += 1
        self.reward1 += reward1
        self.reward2 += reward2
        #print("reward1: ", reward1, "reward2: ", reward2)
        if self.parent:
            self.parent.backpropagate(reward1, reward2)

class MCTS():
    def __init__(self):
        self.root = Node()
    
    def get_action(self, env, n_simulations=6_000):
        #print("input board:", board)
        
        env.board = -env.board
        self.root = Node(env=env)
        #print("Start board when choosing action: ",self.root.env.board)
            
        for _ in range(n_simulations):
            # Select node from root
            node = self.root
            while bool(node.children):
                node = node.select()
            
            # Expand node
            node.expand()
            
            # Simulate root node
            #node = node.select()
            result, board = node.simulate() # board is for debugging purpose
            #print("result: ", result)

            # Backpropagate with simulation result
            if result == 1:
                node.backpropagate(1, 0)
            elif result == -1:
                node.backpropagate(0, 1)
            else:
                node.backpropagate(0, 0)
        #print(self.env.board)
        visits = [child.visits for child in self.root.children]
        visits = np.array(visits)
        ucb_values = [self.root.calculate_UCB(child) for child in self.root.children]
        print("UCB values:", ucb_values)
        print(visits)
        print(visits / np.sum(visits))
        print("length: ", len(visits))
        print("best move?", np.argmax(visits))
        return self.root.children[np.argmax(visits)].action

if __name__ == "__main__":
    
    board = np.zeros((6, 7))
    mcts = MCTS()
    print(mcts.get_action(board))
    
    
    
    