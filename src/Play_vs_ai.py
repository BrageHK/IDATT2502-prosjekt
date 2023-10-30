import pygame
import sys
import torch
import numpy as np

from MCST_basic import MCTS
from ConnectFourEnv import ConnectFour


class ConnectFourPygame:
    
    def __init__(self):
        
        pygame.init()

        self.WIDTH, self.HEIGHT = 700, 600
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.SQUARESIZE = 100
        self.RADIUS = int(self.SQUARESIZE / 2 - 5)

        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        self.turn = 0
        self.running = True

        self.opponent = MCTS()

        pygame.font.init()
        self.font = pygame.font.Font(None, 74)

    def get_board(self):
        return self.board

    def draw_board(self):
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                pygame.draw.rect(self.window, self.BLACK, (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.window, self.BLACK, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)

        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == 1:
                    pygame.draw.circle(self.window, self.RED, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
                elif self.board[r][c] == -1:
                    pygame.draw.circle(self.window, self.YELLOW, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
        pygame.display.update()

    def drop_piece(self, row, col, piece):
        print("Drop piece(row, col, piece): ", row, col, piece)
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[self.ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == 0:
                return r

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
        self.board = [[0] * self.COLUMN_COUNT for r in range(self.ROW_COUNT)]

    def get_move(self, state):
        return self.opponent.get_action(state) # Get the best action

    ### reset
    ### step
    def check_win_and_update_display(self, piece):
        """Check for a win and update the display if needed"""
        if self.winning_move(piece):
            self.draw_board()
            player_number = 1 if piece == 1 else 2
            print(f"Player {player_number} wins!")
            text_surface = self.font.render(f"Player {player_number} wins!", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.window.blit(text_surface, text_rect)
            pygame.display.update()
            pygame.time.wait(3000)
            self.running = False

    def run(self, bot=False):
        """Game loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // self.SQUARESIZE
                    if self.is_valid_location(col):
                        row = self.get_next_open_row(col)
                        piece = -1 if bot else 1
                        self.drop_piece(row, col, piece)
                        self.check_win_and_update_display(piece)
                        self.draw_board()
                        self.turn += 1

                        # AI's turn after the human's turn
                        if bot and self.running:
                            col = self.get_move(self.board)
                            if self.is_valid_location(col):
                                row = self.get_next_open_row(col)
                                self.drop_piece(row, col, 1)
                                self.check_win_and_update_display(1)
                                self.draw_board()
                                self.turn += 1

            pygame.display.update()

if __name__ == "__main__":
    game = ConnectFourPygame()
    game.run(True)
