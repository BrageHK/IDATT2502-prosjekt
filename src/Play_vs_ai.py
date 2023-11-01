import pygame
import torch
import numpy as np
import copy

from Connect_four_env import Connect_four
from MCST_basic import MCTS


class ConnectFourPyGame:
    def __init__(self):
        pygame.init()
        
        self.WIDTH, self.HEIGHT = 700, 600
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Connect Four")
        
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.SQUARESIZE = 100
        self.RADIUS = int(self.SQUARESIZE / 2 - 5)
        
        self.env = Connect_four()
        self.running = True

        self.opponent = MCTS()

        pygame.font.init()
        self.font = pygame.font.Font(None, 74)
        
    def draw_board(self):
        WHITE = (255, 255, 255)  # Define white color

        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                pygame.draw.rect(self.window, self.BLACK, (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.window, self.BLACK, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)

        # Draw white vertical lines between cubes
        for c in range(self.COLUMN_COUNT - 1):
            for r in range(self.ROW_COUNT):
                pygame.draw.line(self.window, WHITE, (c * self.SQUARESIZE + self.SQUARESIZE, r * self.SQUARESIZE), (c * self.SQUARESIZE + self.SQUARESIZE, (r+1) * self.SQUARESIZE), 1)

        # Draw white horizontal lines between cubes
        for r in range(self.ROW_COUNT - 1):
            for c in range(self.COLUMN_COUNT):
                pygame.draw.line(self.window, WHITE, (c * self.SQUARESIZE, (r + 1) * self.SQUARESIZE), ((c + 1) * self.SQUARESIZE, (r + 1) * self.SQUARESIZE), 1)

        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                if self.env.board[r][c] == 1:
                    pygame.draw.circle(self.window, self.RED, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
                elif self.env.board[r][c] == -1:
                    pygame.draw.circle(self.window, self.YELLOW, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
        pygame.display.update()



        
    def play_against_ai(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    quit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // self.SQUARESIZE
                    if self.env.is_valid_location(col):
                        # Player turn
                        self.env.step(col)
                        self.draw_board()
                        
                        if self.env.winning_move(self.env.get_player()):
                            self.running = False
                            pygame.time.wait(3000)
                            pygame.quit()
                            quit()
                            
                        # Ai turn
                        col = self.opponent.get_action(copy.deepcopy(self.env))
                        board, _, _ = self.env.step(col)
                        self.draw_board()
                        print(self.env.board)
                     
                        if self.env.winning_move(self.env.get_player()):
                            self.running = False
                            pygame.time.wait(3000)
                            pygame.quit()
                            quit()

if __name__ == "__main__":
    game = ConnectFourPyGame()
    game.draw_board()
    game.play_against_ai()