import numpy as np
import copy
from ConnectFourEnv import ConnectFour

class Node():
    def __init__(self, parent=None, env=ConnectFour()):
        self.children = []
        self.parent = parent
        self.env = copy.deepcopy(env)
        self.reward1 = 0
        self.reward2 = 0
        self.visits = 0
        
        self.c = 4 # Exploration parameter

    def random_act(self, env):
        legal_moves = [x for x in range(7) if env.is_valid_location(x)]
        return np.random.choice(legal_moves)

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
        
        return np.abs(child.reward1 - child.reward2) / child.visits + np.sqrt(self.c * np.log(self.visits) / child.visits)

    def select(self):
        #print(" are there children? ", bool(self.children))
        if not bool(self.children):
            return self
        UCB_values = []
        #print("children: ", self.children)
        for child in self.children:
            UCB_values.append(self.calculate_UCB(child))
        return self.children[np.argmax(UCB_values)]

    # Adds a child node for untried action and returns it
    def expand(self):
        if self.visits > 0:
            legal_moves = self.env.get_legal_moves()
            print("legal_moves: ", legal_moves)
            for action in legal_moves:
                next_env = copy.deepcopy(self.env)
                _, _, _ = next_env.step(action)
                child = Node(parent=self, env=next_env)
                self.children.append(child)

    # Simulate random actions until a terminal state is reached
    def simulate(self):
        simulated_game = self.env
        board = simulated_game.board
        current_player = simulated_game.get_player() # -1 or 1
        done = False
        _, reward, done = simulated_game.check_game_over(current_player, board)
        
        while not done:
            action = self.random_act(simulated_game)
            next_state, reward, done = simulated_game.step(action)
            board = next_state
        
        winning_player = simulated_game.get_player()
        return reward * winning_player, board # Return 1 if the player won, -1 if the player lost, and 0 if it was a draw.
    
    def backpropagate(self, reward1, reward2):
        self.visits += 1
        self.reward1 += reward1
        self.reward2 += reward2
        if self.parent:
            self.parent.backpropagate(reward1, reward2)

class MCTS():
    def get_action(self, board, n_simulations=10_000):
        self.n_simulations = n_simulations
        #print("input board:", board)
        self.root = Node()
        self.root.env.board = board
            
        for _ in range(self.n_simulations):
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
        print([child.visits for child in self.root.children])
        return np.argmax([child.visits for child in self.root.children])

if __name__ == "__main__":
    
    lol = MCTS()
    print(lol.get_action(np.zeros((6, 7))))
    
    