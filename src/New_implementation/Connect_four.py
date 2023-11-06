import numpy as np
from MCTS import MCTS

class ConnectFour:
    def __init__(self):
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.action_space = self.COLUMN_COUNT
        self.in_a_row = 4
        #self.last_row = None
        #self.last_col = None
        
    def get_initial_state(self):
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
    
    def get_next_open_row(self, state, col):
        for row in range(self.ROW_COUNT):
            if state[row][col] == 0:
                return row
    
    def drop_piece(self, state, col, player):
        row = self.get_next_open_row(state, col)
        state[row][col] = player
        #self.last_row = row  # Add this line
        #self.last_col = col  # Add this line
        
    def check_state_format(self, state):
        return state.size == self.ROW_COUNT * self.COLUMN_COUNT
    
    def check_game_over(self, state, action):
        if self.check_win(state, action):
            return (1, True) # Win
        if np.sum(self.get_valid_moves(state)) == 0:
            return (0, True) # Draw
        return (0, False) # Game goes on
    
    def step(self, state, action, player):
        """
        The agent does an action, and the environment returns the next state, the reward, and whether the game is over.
        The action number corresponds to the column which the piece should be dropped in.
        return: (state, reward, done)
        """
        
        self.drop_piece(state, action, player)

        reward, done = self.check_game_over(state, action)
        
        return state, reward, done
    
    # def get_next_state(self, state, action, player):
    #     row = np.max(np.where(state[:, action] == 0))
    #     state[row, action] = player
    #     return state
    
    def is_valid_location(self, col, state):
        return state[self.ROW_COUNT - 1][col] == 0
    
    def get_valid_moves(self, state):
        legal_moves = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(col, state):
                legal_moves.append(1)
            else:
                legal_moves.append(0)
        return np.array(legal_moves)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.ROW_COUNT
                    or c < 0 
                    or c >= self.COLUMN_COUNT
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state): # TODO: edit
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state
    
    def print_board(self, board): # TODO: should be in benchmark
        inverted_board = np.flipud(board)  # This inverts the y-axis.
        board_str = ""
        for row in inverted_board:
            for cell in row:
                if cell == 1:
                    board_str += " X "
                elif cell == -1:
                    board_str += " O "
                else:
                    board_str += " . "  # Assuming 0 is an empty cell, we replace it with a dot.
            board_str += "\n"
        print(board_str)
    
if __name__ == "__main__":
    game = ConnectFour()
    player = 1
    mcts = MCTS(game, num_iterations=10_000)
    
    state = game.get_initial_state()


    while True:
        game.print_board(state)
        
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(game.action_space) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue
                
        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            
        state, reward, done = game.step(state, action, player)
        
        value, is_terminal = game.check_game_over(state, action)
        
        if is_terminal:
            game.print_board(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
            
        player = game.get_opponent(player)