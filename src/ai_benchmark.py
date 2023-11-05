import time
import logging
from Connect_four_env import ConnectFour
from MCTS.MCTS import MCTS
from Node.NodeType import NodeType
import json
import numpy as np
import random

from multiprocessing import Pool, cpu_count

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


def play_game(mcts1, mcts2, match_id, name1, name2):
    logging.info(f'Starting match {match_id} between {name1} and {name2}')
    env = ConnectFour()
    time_stats = {name1: 0, name2: 0}
    actions_stats = {name1: 0, name2: 0}
    players = {1: (mcts1, name1),
                -1: (mcts2, name2)}
    winner = None 
    done = False
    reward = None
    current_player = 1
    # try:
    while not done:
        mcts, player_name = players[current_player]
        #print(f"player {player_name}'s turn ")
        start_time = time.time()
        action = mcts.get_action(env.deepcopy())
        time_stats[player_name] += time.time() - start_time
        actions_stats[player_name] += 1
        print(f"Player {player_name} chose action {action}")
        reward, done = env.step(action)
        print_board(env.board)
        current_player = -current_player
    winner = winner = -current_player if reward == 1 else (0 if env.turn == 42 else current_player)
    # except Exception as e:
    #     logging.error(f"Error during match {match_id}: {e}")
    
    logging.info(f'Match {match_id} finished. player {winner} won  Winner: {players[winner][1] if winner else "Draw"}')
    
    return winner, time_stats, actions_stats, match_id, name1, name2


def benchmark_mcts(mcts_versions, num_games=20):
    results = {}
    match_id = 1

    args_list = []
    for _ in range(num_games // 2):
        for name1, mcts1 in mcts_versions.items():
            for name2, mcts2 in mcts_versions.items():
                if name1 != name2:
                    args = (mcts1, mcts2, match_id, name1, name2)
                    args_list.append(args)
                    match_id += 1

    with Pool(1) as pool:
    #with Pool(cpu_count()) as pool:
        results_list = pool.starmap(play_game, args_list)

    for result in results_list:
        winner, game_time_stats, game_actions_stats, match_id, name1, name2 = result
        matchup = tuple(sorted([name1, name2]))
        matchup_str = str(matchup)
        if matchup_str not in results:
            results[matchup_str] = {
                name1: {"Wins": 0, "Losses": 0, "Draws": 0, "TotalTime": 0, "TotalActions": 0, "GamesPlayed": 0},
                name2: {"Wins": 0, "Losses": 0, "Draws": 0, "TotalTime": 0, "TotalActions": 0, "GamesPlayed": 0}
        }

        results = update_stats(
            results,
            name1, name2, winner, 
            game_time_stats, game_actions_stats, 
            matchup_str  # Pass the string version of the matchup
        )

    notify_benchmark_finished()
    return results



def update_stats(results, name1, name2, winner, game_time_stats, game_actions_stats, matchup):
    matchup_str = str(matchup)
    # Ensure the matchup entry exists with all necessary sub-structures
    if matchup_str not in results:
        results[matchup_str] = {
            name1: {"Wins": 0, "Losses": 0, "Draws": 0, "TotalTime": 0, "TotalActions": 0, "GamesPlayed": 0},
            name2: {"Wins": 0, "Losses": 0, "Draws": 0, "TotalTime": 0, "TotalActions": 0, "GamesPlayed": 0}
        }
        
    # Update the win/loss/draw counts
    if winner == 1:
        results[matchup_str][name1]["Wins"] += 1
        results[matchup_str][name2]["Losses"] += 1
    elif winner == -1:
        results[matchup_str][name1]["Losses"] += 1
        results[matchup_str][name2]["Wins"] += 1
    else:
        results[matchup_str][name1]["Draws"] += 1
        results[matchup_str][name2]["Draws"] += 1
        
    # Aggregate time and action statistics
    for name in [name1, name2]:
        results[matchup_str][name]["TotalTime"] += game_time_stats[name]
        results[matchup_str][name]["TotalActions"] += game_actions_stats[name]
        results[matchup_str][name]["GamesPlayed"] += 1

    # Calculate averages
    for name in [name1, name2]:
        games_played = results[matchup_str][name]["GamesPlayed"]
        if games_played > 0:  # Avoid division by zero
            results[matchup_str][name]["AvgTimePerMatch"] = results[matchup_str][name]["TotalTime"] / games_played
            results[matchup_str][name]["AvgActionsPerMatch"] = results[matchup_str][name]["TotalActions"] / games_played
            results[matchup_str][name]["AvgTimePerAction"] = (results[matchup_str][name]["TotalTime"] / results[matchup_str][name]["TotalActions"]) if results[matchup_str][name]["TotalActions"] > 0 else 0

    return results


class Randomf():
    def get_action(self, env):
        return random.choice(env.get_legal_moves())


if __name__ == "__main__":

    mcts_versions = {
    #"1": MCTS(n_simulations=50000), # "genious
    #"Basic MCTS": MCTS(n_simulations=20_000),
    #"dumbass": MCTS(n_simulations=10_000),
    "smart" :MCTS(num_simulations=8),

    "random" :Randomf(),
    #"dumb" :MCTS(n_simulations=10),
    
    # Add other versions here
    }

    results = benchmark_mcts(mcts_versions, num_games=2)
    
    # Prepare data for writing to JSON
    data_to_write = {
        "Results": results,
    }
        
    # Write results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(data_to_write, f, indent=4)
        
    print("Results have been written to results.json")


