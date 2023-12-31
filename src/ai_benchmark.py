import time
import logging
from Connect_four_env import ConnectFour
from MCTS.MCTS import MCTS
from Node.NodeType import NodeType
from NeuralNet import AlphaPredictorNerualNet
import json
import numpy as np
import random
import torch

import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from plyer import notification

# Configure logging
logging.basicConfig(level=logging.INFO)

def print_board(board):
    inverted_board = np.flipud(board)  # This inverts the y-axis.
    board_str = ""
    for row in inverted_board:
        for cell in row:
            if cell == 1:
                board_str += " X "
            elif cell == -1:
                board_str += " O "
            else:
                board_str += " . "  # Assuming 0 is an empty cell, we replace it with a dot.
        board_str += "\n"
    print(board_str)

def notify_benchmark_finished():
    """Sends a system notification that benchmarking is finished."""
    try:
        notification.notify(
            title="Benchmark finished",
            message="The benchmark process has been completed successfully.",
            app_name="Benchmark Tool",
            timeout=10  # Notification duration in seconds
        )
    except Exception as e:
        logging.error(f"Failed to send notification: {e}")

def play_game(env, mcts1, mcts2, match_id, name1, name2):
    logging.info(f'Starting match {match_id} between {name1} and {name2}')
    state = env.get_initial_state()
    time_stats = {name1: 0, name2: 0}
    time_per_action = {name1: [], name2: []}  # Initialize dictionary to store time per action
    actions_stats = {name1: 0, name2: 0}
    players = {1: (mcts1, name1),
                -1: (mcts2, name2)}
    winner = None 
    done = False
    reward = None
    current_player = 1
    turn = 0
    try:
        while not done:
            mcts, player_name = players[current_player]
            #print(f"player {player_name}'s turn ")
            neutral_state = env.change_perspective(state, current_player)
            start_time = time.time()
            action = mcts.search(neutral_state)
            #mcts_probs = mcts.search(neutral_state)
            action_time = time.time() - start_time
            time_stats[player_name] += action_time
            #action = np.argmax(mcts_probs)
            time_per_action[player_name].append(action_time)  # Record time for this action
            actions_stats[player_name] += 1
            #print(f"Player {player_name} chose action {action}")
            turn += 1
            state ,reward, done = env.step(state ,action, current_player)
            #print_board(state)
            #print_board(state)
            current_player = -current_player

        print_board(state)

        if turn == 42 and reward == 0:
            winner = 0
        else:
            winner = -current_player if reward == 1 else current_player

    except Exception as e:
        logging.error(f"Error during match {match_id}: {e}")
    
    if winner == 0:
        logging.info(f'Match {match_id} finished. Draw.')
    elif winner in (1, -1):
        logging.info(f'Match {match_id} finished. Player {winner} won. Winner: {players[winner][1]}')
    
    
    return winner, time_stats, actions_stats, time_per_action, match_id, name1, name2


def benchmark_mcts( mcts_versions, env=ConnectFour(), num_games=20):
    results = {}
    match_id = 1

    args_list = []
    for _ in range(num_games // 2):
        for name1, mcts1 in mcts_versions.items():
            for name2, mcts2 in mcts_versions.items():
                if name1 != name2:
                    args = (env, mcts1, mcts2, match_id, name1, name2)
                    args_list.append(args)
                    match_id += 1

    #with Pool(1) as pool:
    with mp.Pool(mp.cpu_count()) as pool:
        results_list = pool.starmap(play_game, args_list)

    for result in results_list:
        winner, game_time_stats, game_actions_stats, game_time_per_action, match_id, name1, name2 = result
        matchup = tuple(sorted([name1, name2]))
        matchup_str = str(matchup)

        results = update_stats(
            results, name1, name2, winner, 
            game_time_stats, game_actions_stats, game_time_per_action,
            match_id,
            matchup_str
        )

    notify_benchmark_finished()
    return results



def update_stats(results, name1, name2, winner, game_time_stats, game_actions_stats, game_time_per_action, match_id, matchup):
    matchup_str = str(matchup)
    # Ensure the matchup entry exists with all necessary sub-structures
    if matchup_str not in results:
        results[matchup_str] = {
            name1: {
                "Wins": 0, "Losses": 0, "Draws": 0, 
                "TotalTime": 0, "TotalActions": 0, 
                "GamesPlayed": 0, "FirstPlayerWins": 0,
                "TimePerAction": []  # Added "FirstPlayerWins" here
            },
            name2: {
                "Wins": 0, "Losses": 0, "Draws": 0, 
                "TotalTime": 0, "TotalActions": 0, 
                "GamesPlayed": 0, "FirstPlayerWins": 0,
                "TimePerAction": []  # And here
            }
        }

    # Update the win/loss/draw counts and "FirstPlayerWins"
    if winner == 1:
        results[matchup_str][name1]["Wins"] += 1
        results[matchup_str][name2]["Losses"] += 1
        results[matchup_str][name1]["FirstPlayerWins"] += 1  # Increment if first player wins
    elif winner == -1:
        results[matchup_str][name1]["Losses"] += 1
        results[matchup_str][name2]["Wins"] += 1
    elif winner == 0:
        results[matchup_str][name1]["Draws"] += 1
        results[matchup_str][name2]["Draws"] += 1
        
    # Aggregate time and action statistics
    for name in [name1, name2]:
        results[matchup_str][name]["TotalTime"] += game_time_stats[name]
        results[matchup_str][name]["TotalActions"] += game_actions_stats[name]
        results[matchup_str][name]["GamesPlayed"] += 1
        # Append time data for the current match
        results[matchup_str][name]["TimePerAction"].append(game_time_per_action[name])


    # Calculate averages
    for name in [name1, name2]:
        games_played = results[matchup_str][name]["GamesPlayed"]
        if games_played > 0:  # Avoid division by zero
            results[matchup_str][name]["AvgTimePerMatch"] = results[matchup_str][name]["TotalTime"] / games_played
            results[matchup_str][name]["AvgActionsPerMatch"] = results[matchup_str][name]["TotalActions"] / games_played
            results[matchup_str][name]["AvgTimePerAction"] = (results[matchup_str][name]["TotalTime"] / results[matchup_str][name]["TotalActions"]) if results[matchup_str][name]["TotalActions"] > 0 else 0

    return results



class Randomf():
    def __init__(self, env):
        self.env = env
    def search(self, state):
        return random.choice(self.env.get_legal_moves(state))


def get_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaPredictorNerualNet(num_resBlocks=9, env=ConnectFour(), device=device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

if __name__ == "__main__":
    env = ConnectFour()
    model100 = get_model("data/eilor/model100.pt")
    model364 = get_model("data/eilor/model364.pt")
    mcts_versions = {
    # "genious normalized": MCTS(env, num_iterations=10_000, NODE_TYPE=NodeType.NODE_NORMALIZED), # "genious
    # "genious ": MCTS(env, num_iterations=10_000, NODE_TYPE=NodeType.NODE), # "genious
    # "Basic MCTS ": MCTS(env, num_iterations=5_000, NODE_TYPE=NodeType.NODE),
    # "Basic normalized ": MCTS(env, num_iterations=5_000, NODE_TYPE=NodeType.NODE_NORMALIZED),
    #"dumbass": MCTS(n_simulations=10_000),
    #"smart" :MCTS(env=ConnectFour() ,num_iterations= 20000),

    #"random" :Randomf(env),
    #"dumb" :MCTS(n_simulations=10),
    
    # Add other versions here
    "100": MCTS(env, 600, NodeType.NODE_NN, model100),
    "364": MCTS(env, 600, NodeType.NODE_NN, model364)
    }

    results = benchmark_mcts(mcts_versions, env, num_games=mp.cpu_count())
    
    # Prepare data for writing to JSON
    data_to_write = {
        "Results": results,
    }
        
    # Write results to a JSON file
    with open('data/results.json', 'w') as f:
        json.dump(data_to_write, f, indent=4)
        
    print("Results have been written to data/results.json")


