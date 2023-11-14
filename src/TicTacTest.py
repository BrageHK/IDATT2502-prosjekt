from TicTacToe import TicTacToe
from NeuralNet import AlphaPredictorNerualNet
import torch

env = TicTacToe()
board = env.get_initial_state()

print(env.__repr__())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("data/TicTacToe/model.pt", map_location=device)

done = 0
while not done:
    print(board)
    action = input("Enter action: ")
    board, reward, done = env.step(action)
    