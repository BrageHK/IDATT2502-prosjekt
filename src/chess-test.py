import gym
import gym_chess
import chess
import random


env = gym.make('ChessAlphaZero-v0')
print(env.render(mode='unicode'), "\n")
env.reset()
print(env.legal_actions, "\n")
move = chess.Move.from_uci('e2e4')
print(env.encode(move))
print(env.encode(move) in env.legal_actions)

done = False
move_num = 0
state_info = None
while not done:
    action = random.sample(env.legal_actions, 1)[0]
    state_info = env.step(action)
    #print("Done: ", done, "Action: ", action)
    done = state_info[2]
    print(env.render(mode='unicode'), "\n")
    move_num += 1
    print("Move number:", move_num)
env.close()