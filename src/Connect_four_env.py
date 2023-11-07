import numpy as np
from enum import Enum

class BoardState(Enum):
    PLAYER_1 = 1
    AVAILABLE = 0
    OPPONENT = -1

class ConnectFour:
    def __init__(self):
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.action_space = self.COLUMN_COUNT
        self.in_a_row = 4
        #self.last_row = None
        #self.last_col = None
        
    def get_initial_state(self): # TODO: remove - it is no point whith this, can create a new environment
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
    
    def drop_piece(self, state, col, player):
        row = self.get_next_open_row(state, col)
        state[row][col] = player
        #self.last_row = row  # Add this line
        #self.last_col = col  # Add this line
      
    def is_valid_location(self, col, state):
        return state[self.ROW_COUNT - 1][col] == 0
    
    def get_next_open_row(self, state, col):
        for row in range(self.ROW_COUNT):
            if state[row][col] == 0:
                return row
    
    def get_legal_moves(self, state):
        legal_moves = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(col, state):
                legal_moves.append(col)
        return legal_moves
    
    def get_legal_moves_bool_array(self, state):
        legal_moves = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(col, state):
                legal_moves.append(1)
            else:
                legal_moves.append(0)
        return np.array(legal_moves)

    def check_win(self, state, action): # TODO: rewrite to be understandable
        if action == None:
            return False
        
        row = np.max(np.where(state[:, action] != 0))
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
    
    def check_state_format(self, state):
        return state.size == self.ROW_COUNT * self.COLUMN_COUNT
    
    # TODO: rewrite code to automatic return opponent player and opponent value?
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player
    
    def check_game_over(self, state, action):
        if self.check_win(state, action):
            return (1, True) # Win
        if np.sum(self.get_legal_moves_bool_array(state)) == 0:
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
    
    def get_encoded_state(self, state):
        player_mask = [state == board_state.value for board_state in BoardState]
        encoded_state = np.stack(player_mask).astype(np.float32)
        return encoded_state
