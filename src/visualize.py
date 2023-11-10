import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from JSON file
with open('src/data/results.json', 'r') as file:
    data = json.load(file)


# Extract the first matchup data for simplicity
matchup_data = list(data["Results"].values())[0]

# Data preparation
players = list(matchup_data.keys())
matchup_name = ' vs '.join(players)  # Create a matchup name
wins = [matchup_data[player]["Wins"] for player in players]
first_player_wins = [matchup_data[player]["FirstPlayerWins"] for player in players]
total_games = matchup_data[players[0]]["GamesPlayed"]  # Assuming the number of games is the same for both
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
rects1 = ax.bar(x, first_player_wins, width, label='First Player Wins', color='green')
rects2 = ax.bar(x, second_player_wins, width, label='Second Player Wins', color='lightgreen', bottom=first_player_wins)

# Draw bars are in their own category, placed right after the second player bar
rects3 = ax.bar(draw_bar_x, draws, width, label='Draws', color='blue')

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Counts')
ax.set_title(f'{matchup_name} - Total Games: {total_games}')
ax.set_xticks(np.append(x, draw_bar_x))
ax.set_xticklabels(players + ['Draws'])
ax.legend()

def add_value_labels(ax, rects, data, total, is_total=False, offset_x=0, offset_y=1):
    for rect, datum in zip(rects, data):
        percentage = f'({datum / total:.1%})' if total > 0 else ''
        label = f'{datum} {percentage}'
        height = rect.get_height()
        if is_total:
            # If this is the total wins label, we want it above the second player wins bar
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height + first_player_wins[rects.index(rect)]),
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
add_value_labels(ax, rects1, first_player_wins, total_games, offset_x=width, offset_y=2)  # Add labels for 'First Player Wins'
add_value_labels(ax, rects2, second_player_wins, total_games, offset_x=width, offset_y=0.625)  # Offset for 'Second Player Wins'
add_value_labels(ax, rects2, wins, total_games, is_total=True)  # Add labels for 'Total Wins', above second player wins
add_value_labels(ax, rects3, [draws], total_games)  # Add label for 'Draws'
# Adjust the layout
plt.tight_layout()

# Save the plot as SVG
plt.savefig('matchup_graph.svg', format='svg')
