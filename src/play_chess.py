import gym
import gym_chess
import chess
import numpy as np
from MCTS.MCTS import MCTS
from Node.NodeType import NodeType
from NeuralNetChess import AlphaPredictorNerualNetChess
import torch
import copy

done = False
env = gym.make("ChessAlphaZero-v0")
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaPredictorNerualNetChess(20, device=device)
model.load_state_dict(
    torch.load(
        "data/NODE_NN_CHESS/<MoveEncoding<BoardEncoding<Chess<ChessAlphaZero-v0>>>>/model/model.pt",
        map_location=device,
    )
)
model.eval()
mcts = MCTS(env, 200, NODE_TYPE=NodeType.NODE_NN_CHESS, model=model)


def ai_move(env):
    action = mcts.search(copy.deepcopy(env), training=False)
    print("action: ", action)
    state_info = env.step(action)
    print(env.render(mode="unicode"), "\n")
    return state_info[2]


while not done:
    # Computer turn
    done = ai_move(env)
    if done:
        break





    # Your turn
    # print(env.render(mode="unicode"), "\n")
    # moved = False
    # move = None
    # while not moved:
    #     move = input(f"make a move from the legal moves here {env.legal_moves}:")
    #     encoded_move = env.encode(move)
    #     if encoded_move in env.legal_actions:
    #         moved = True
    # state_info = env.step(move)

print(env.render(mode="unicode"), "\n")
