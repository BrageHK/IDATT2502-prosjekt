import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data from JSON file
with open('data/results.json', 'r') as file:
    data = json.load(file)

# Loop through all matchups in the data
for matchup_name, matchup_data in data["Results"].items():
    # Data preparation
    players = list(matchup_data.keys())
    wins = [matchup_data[player]["Wins"] for player in players]
    first_player_wins = [matchup_data[player]["FirstPlayerWins"] for player in players]
    total_games = [matchup_data[player]["GamesPlayed"] for player in players]  # Assuming the number of games is the same for both
    draws = matchup_data[players[0]]["Draws"]  # Assuming draws are the same for both

    # Calculate wins that were not first player wins
    second_player_wins = [wins[i] - first_player_wins[i] for i in range(len(wins))]

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # The x position of bars for players
    x = np.arange(len(players))  # [0, 1]
    width = 0.35  # Width of the player bars

    # Calculate the x position for the draws bar, which will be right after the player bars
    draw_bar_x = x[-1] + 1  # Set draw bar next to the player bars

    # Stack 'First Player Wins' on top of 'Second Player Wins'
    rects1 = ax.bar(x, first_player_wins, width, label='Winner When Starting', color='green')
    rects2 = ax.bar(x, second_player_wins, width, label='Winner When Second', color='lightgreen', bottom=first_player_wins)

    # Draw bars are in their own category, placed right after the second player bar
    rects3 = ax.bar(draw_bar_x, draws, width, label='Draws', color='blue')

    # Adding labels, title, and custom x-axis tick labels
    player_labels = [f'Wins {player}' for player in players]
    ax.set_ylabel('matches')
    ax.set_title(f'{players[0]} vs {players[1]} - Total Games: {total_games[0]}')
    ax.set_xticks(np.append(x, draw_bar_x))
    ax.set_xticklabels(player_labels + ['Draws'])
    ax.legend()

    def add_value_labels(ax, rects, data, total, is_total=False, offset_x=0, offset_y=1):
        for rect, datum in zip(rects, data):
            percentage = f'({datum / total[rects.index(rect)]:.1%})' if total[rects.index(rect)] > 0 else ''
            #percentage = f'({datum / total:.1%})' if total > 0 else ''
            label = f'{datum} {percentage}'
            height = rect.get_height()
            if is_total:
                # If this is the total wins label, we want it above the second player wins bar
                ax.annotate(label,
                            xy=(rect.get_x() + rect.get_width() / 2 + offset_x, (height/ offset_y ) + first_player_wins[rects.index(rect)]),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            else:
                # For other bars, just display the label at the top
                ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2 + offset_x, height / offset_y),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Call the function to add labels to the bars
    add_value_labels(ax, rects1, first_player_wins, wins, offset_x=width, offset_y=2)  # Add labels for 'First Player Wins'
    add_value_labels(ax, rects2, second_player_wins, wins, is_total=True, offset_x=width, offset_y=2)  # Offset for 'Second Player Wins'
    add_value_labels(ax, rects2, wins, total_games, is_total=True)  # Add labels for 'Total Wins', above second player wins
    add_value_labels(ax, rects3, [draws], total_games)  # Add label for 'Draws'
    # Adjust the layout
    plt.tight_layout()

    # Create the plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the plot as SVG
    plt.savefig(f'plots/{players[0].replace(" ", "_")}_vs_{players[1].replace(" ", "_")}.svg', format='svg')

    
    # Close the plot to avoid memory leaks
    plt.close()
