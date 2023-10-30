import numpy as np

class ConnectFour:
    
    def __init__(self):
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.turn = 0
        self.action_space = 7
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))

    def drop_piece(self, col, piece):
        row = self.get_next_open_row(col)
        self.board[row][col] = piece

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

    def reset(self):
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        self.turn = 0
        return self.board.flatten()
    
    def get_player(self):
        return -1 if self.turn % 2 == 0 else 1
    
    def check_game_over(self, piece):
        if self.winning_move(piece):
            return (1, True) # Win
        if self.turn == 41: #TODO: WHAT?? 39???? hææææ?? 
            return (0, True) # Draw
        else:
            return (0, False) # Game goes on

    def step(self, action):
        """
        The agent does an action, and the environment returns the next state, the reward, and whether the game is over.
        The action number corresponds to the column which the piece should be dropped in.
        return: (next_state, reward, done)
        """
        
        self.drop_piece(action, self.get_player())
        self.turn += 1
        outputBoard = self.board
        reward, done = self.check_game_over(self.get_player())
        return outputBoard, reward, done
        
        