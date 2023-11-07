import pygame
import torch
import numpy as np
import copy

import time 
import curses
from Connect_four_env import ConnectFour

from MCTS.MCTS import MCTS
from Node.NodeType import NodeType

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
        
        self.env = ConnectFour()
        self.running = True

        self.opponent = MCTS(NODE_TYPE=NodeType.NODE_SINGLE, num_simualtions=1000)

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
                        col = self.opponent.get_action(self.env)
                        board, _, _ = self.env.step(col)
                        self.draw_board()
                        print(self.env.board)
                     
                        if self.env.winning_move(self.env.get_player()):
                            self.running = False
                            pygame.time.wait(3000)
                            pygame.quit()
                            quit()


class ConnectFourTerminal:

    def __init__(self, stdscr, opponent_algorithm):
        self.env = ConnectFour()
        self.state = self.env.get_initial_state()
        self.done = False
        self.current_player = 1
        self.result = None
        self.turn = 1
        self.opponent = opponent_algorithm  # Use the passed algorithm
        self.cursor_x = 0
        self.stdscr = stdscr
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    @classmethod
    def initialize_terminal(cls, stdscr, opponent_algorithm):
        game_instance = cls(stdscr, opponent_algorithm)
        game_instance.play_against_ai()

    def draw_board(self, message=None):
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "\nCurrent Board:")
        self.stdscr.addstr(1, 0, "---------------")
        row_num = 2
        for row in reversed(self.state):
            self.stdscr.addstr(row_num, 0, "|")
            for cell in row:
                if cell == 1:
                    self.stdscr.addstr("X", curses.color_pair(1))
                elif cell == -1:
                    self.stdscr.addstr("O", curses.color_pair(2))
                else:
                    self.stdscr.addstr(".")
                self.stdscr.addstr("|")
            row_num += 1
        self.stdscr.addstr(row_num, 0, "---------------")
        self.stdscr.addstr(row_num + 1, 0, " 0 1 2 3 4 5 6")
        # Clear previous message with spaces (assuming max message length 16 chars)
        self.stdscr.addstr(row_num + 2, 0, " " * 16)
        
        # Only draw cursor if there's no message
        if message is None:
            self.stdscr.addstr(row_num + 2, self.cursor_x * 2 + 1, "^")
        else:
            self.stdscr.addstr(row_num + 2, 0, message)
            
        self.stdscr.refresh()

    
    def choose_who_starts(self):
        self.stdscr.addstr(0, 0, "Who starts? (p=Player, a=AI, q=Quit): ")
        self.stdscr.refresh()
        while True:
            choice = self.stdscr.getch()
            if choice == ord('p'):
                self.player_starts = True
                break
            elif choice == ord('a'):
                self.player_starts = False
                break
            elif choice == ord('q'):
                self.player_starts = True
                self.done = True
                return

    def ai_turn(self):
        self.draw_board("Waiting for AI...")
        neutral_state = self.env.change_perspective(self.state, self.current_player)
        action = self.opponent.search(neutral_state)
        self.state, self.result , self.done = self.env.step(self.state,action, self.current_player)
        self.current_player = -self.current_player 
        self.turn += 1


    def player_turn(self):
        turn_completed = False
        while not turn_completed and not self.done:
            self.draw_board()
            key = self.stdscr.getch()

            if key in [curses.KEY_LEFT, ord('h')]:
                self.cursor_x = max(0, self.cursor_x - 1)
            elif key in [curses.KEY_RIGHT, ord('l')]:
                self.cursor_x = min(6, self.cursor_x + 1)
            elif key in [curses.KEY_ENTER, 10, 13, ord(' ')]:
                if self.env.is_valid_location(self.cursor_x, self.state):
                    self.state, self.result, self.done =self.env.step(self.state, self.cursor_x, self.current_player) 
                    self.current_player = -self.current_player
                    self.turn += 1
                    turn_completed = True  # The turn is completed after a valid move.
                    
            elif key == ord('q'):
                self.done = True
                turn_completed = True

    def play_against_ai(self):
        self.choose_who_starts()

        if not self.player_starts:
            self.ai_turn()

        while not self.done:
            self.player_turn()
            if not self.done:
                self.ai_turn()

        if self.turn == 42 and self.result == 0:
            winner_message = "It's a draw!"
        else:
            # Determine who the winner is
            is_player_winner = (self.player_starts and self.current_player == -1) or (not self.player_starts and self.current_player == 1)
            winner_message = "Player wins!" if is_player_winner else "AI wins!"

        self.draw_board(winner_message)
        time.sleep(5)


if __name__ == "__main__":
    #game = ConnectFourPyGame()
    #game.draw_board()
    #game.play_against_ai()


    #opponent_algorithm = MCTS(NODE_TYPE=NodeType.NODE_DOUBLE, num_simulations=50_000)
    opponent_algorithm = MCTS(env=ConnectFour(), num_iterations=100)



    curses.wrapper(lambda stdscr: ConnectFourTerminal.initialize_terminal(stdscr, opponent_algorithm))
