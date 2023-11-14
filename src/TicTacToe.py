import numpy as np

class TicTacToe:
    def __init__(self):
        self.ROW_COUNT = 3
        self.COLUMN_COUNT = 3
        self.action_space = self.ROW_COUNT * self.COLUMN_COUNT
        
    def __repr__(self):
        return "TicTacToe"
        
    def get_initial_state(self):
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
    
    def get_legal_moves_bool_array(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.COLUMN_COUNT
        column = action % self.COLUMN_COUNT
        player = state[row, column]
        if(player == 0):
            raise ValueError("Trying to check win for empty cell")
        
        return (
            np.sum(state[row, :]) == player * self.COLUMN_COUNT
            or np.sum(state[:, column]) == player * self.ROW_COUNT
            or np.sum(np.diag(state)) == player * self.ROW_COUNT
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.ROW_COUNT
        )
        
    def check_game_over(self, state, action):
        """
        (reward, done)
        """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_legal_moves_bool_array(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
    def check_state_format(self, state):
        return state.size == self.ROW_COUNT * self.COLUMN_COUNT
    
    def step(self, state, action, player):  
        state = state.copy()
        row = action // (self.COLUMN_COUNT)
        column = action % self.COLUMN_COUNT
        state[row, column] = player
        
        reward, done = self.check_game_over(state, action)
        
        return state, reward, done