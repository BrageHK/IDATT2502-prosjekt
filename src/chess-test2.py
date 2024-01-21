import gym
import gym_chess
import chess
import random
import copy
import numpy as np

env = gym.make("ChessAlphaZero-v0")
print(env.render(mode="unicode"), "\n")
env.reset()
print("DONE: ", env.done)

print(env.legal_actions, "\n")
move = chess.Move.from_uci("h7h5")
print("encoded move:", env.encode(move))
print(env.encode(move) in env.legal_actions)
old_board = env.observation(env.env.env._board)

print("decoded: ", env.decode(1022))

env2 = env
done = False
move_num = 0
state_info = None
while not done:
    action = random.sample(env.legal_actions, 1)[0]
    state_info = env.step(action)

    # print("action: ", (action))
    # print("Done: ", done, "Action: ", action)
    done = state_info[2]
    # print(env.render(mode='unicode'), "\n")
    move_num += 1
    # print("Move number:", move_num)
print("Encoded board state: ", state_info[0].shape)
print("Winner: ", state_info[1])
print("Current board: ", env.observation(env.env.env._board).shape)
print(env2 == env)

print(np.array_equal(old_board, env.observation(env.env.env._board)))

test = env.get_observation()
test2 = env.get_observation()

print("test2: ", np.array_equal(test2, env.observation(env.env.env._board)))
print("test: ", np.array_equal(test, env.get_observation()))
env.close()


# import chess.pgn
# pgn = chess.pgn.Game.from_board(env.env.env._board)
# pgn.headers["Event"] = "Some event"
# pgn.headers["Site"] = "Some location"
# pgn.headers["White"] = "White Player"
# pgn.headers["Black"] = "Black Player"
# print(pgn)


