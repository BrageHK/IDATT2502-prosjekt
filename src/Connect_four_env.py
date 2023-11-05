import numpy as np
from enum import Enum
import copy

class BoardState(Enum):
    PLAYER_1 = 1
    AVAILABLE = 0
    OPPONENT = -1

class ConnectFour:
    
    def __init__(self):
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.turn = 1
        self.action_space = 7
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        self.last_col = None  
        self.last_row = None  
    
    def deepcopy(self):
        return copy.deepcopy(self)

    def drop_piece(self, col, piece):
        row = self.get_next_open_row(col)
        self.board[row][col] = piece
        self.last_row = row  # Add this line
        self.last_col = col  # Add this line

    def is_valid_location(self, col):
        return self.board[self.ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == 0:
                return r
            
    def get_legal_moves(self):
        legal_moves = []
        for col in range(self.COLUMN_COUNT):
            if self.is_valid_location(col):
                legal_moves.append(col)
        return legal_moves
    
    def get_legal_moves_bool_array(self):
        legal_moves = []
        for col in range(self.COLUMN_COUNT):
            legal_moves.append(self.is_valid_location(col))
        return legal_moves

    def winning_move(self, piece):
        # Check horizontal locations for win
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True
                
        #if self.last_row is None or self.last_col is None:
        #    print("ligma")
        #   return False
        
        # # directions: (row_increment, col_increment)
        # directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # for row_inc, col_inc in directions:
        #     count = 1  # Start with the last placed piece
            
        #     # Check both directions from the last placed piece
        #     for direction in [-1, 1]:
                
        #         for step in range(1, 4):  # Check next 3 cells
        #             temp_row = self.last_row + row_inc * step * direction  # Reset temp_row
        #             temp_col = self.last_col + col_inc * step * direction  # Reset temp_col
                    
        #             # Check boundaries
        #             if 0 <= temp_row < self.ROW_COUNT and 0 <= temp_col < self.COLUMN_COUNT:
        #                 if self.board[temp_row][temp_col] == piece:
        #                     count += 1
        #                     if count == 4:
        #                         return True
        #                 else:
        #                     break  # No need to check further in this direction
        #             else:
        #                 break  # Out of bounds

        # return False

    def reset(self):
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        self.turn = 1
        return self.board.flatten()
    
    def get_player(self):
        return -1 if self.turn % 2 == 0 else 1
    
    def get_last_player(self):
        return -self.get_player()
    
    def check_game_over(self, piece):
        if self.winning_move(piece):
            return (1, True) # Win
        if self.turn == 42:
            return (0, True) # Draw
        return (0, False) # Game goes on

    def step(self, action):
        """
        The agent does an action, and the environment returns the next state, the reward, and whether the game is over.
        The action number corresponds to the column which the piece should be dropped in.
        return: (reward, done)
        """
        
        player = self.get_player()
        
        self.drop_piece(action, player)
        
        if self.turn < 7:
            self.turn += 1
            return 0, False  # No one can win before 7 moves

        reward, done = self.check_game_over(player)
        self.turn += 1
        
        return reward, done
    
    def get_encoded_state(self, board=None):
        if board is None:
            board = self.board

        player_mask = [board == state.value for state in BoardState]
        encoded_state = np.stack(player_mask).astype(np.float32)
        return encoded_state
        
    def invert_board_for_second_player(self):
        if self.get_player() == -1:
            self.board = -self.board
        