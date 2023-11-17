from TicTacToe import TicTacToe
from NeuralNet import AlphaPredictorNerualNet
from Node.NodeType import NodeType
import MCTS.MCTS as MCTS
import torch
import numpy as np

env = TicTacToe()

print(env.__repr__())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_resBlocks = 4
model = AlphaPredictorNerualNet(num_resBlocks=num_resBlocks, env=env, device=device)
model.load_state_dict(torch.load(f"data/TicTacToe/model-{num_resBlocks}.pt", map_location=device))
mcts = MCTS.MCTS(env, 60, NODE_TYPE=NodeType.NODE_NN, model=model)

done = 0
player = -1
state = env.get_initial_state()

while True:
    print(state)
    
    if player == 1:
        valid_moves = env.get_legal_moves_bool_array(state)
        print("valid_moves", [i for i in range(env.action_space) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
    else:
        neutral_state = env.change_perspective(state, player)
        action = mcts.search(neutral_state)        
        
    state, value, is_terminal = env.step(state, action, player)
        
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = env.get_opponent(player)