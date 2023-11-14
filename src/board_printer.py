from Connect_four_env import ConnectFour
import numpy as np

class BoardPrinter:
    def __init__():
        pass
    
    def print_board(board):
        player_1_matrix = np.flip(np.array(board[0]))
        player_2_matrix = np.flip(np.array(board[2]))
        availilable_moves = np.flip(np.array(board[1]))
        
        for i in range(6):
            for j in range(7):
                if player_1_matrix[i][j] == 1:
                    print("X", end=" ")
                elif player_2_matrix[i][j] == 1:
                    print("O", end=" ")
                elif availilable_moves[i][j] == 1:
                    print("-", end=" ")
                else:
                    raise ValueError("Invalid board")
            print()
            
    def memory_debugger(result_list):
        for i in range(len(result_list)):
            print("Position ", i)
            BoardPrinter.print_board(result_list[i][0])
            print("MCTS prob: ", result_list[i][1])
            print("Outcome: ", result_list[i][2])
            print()