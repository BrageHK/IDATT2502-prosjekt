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
    timePerAction = [matchup_data[player]["TimePerAction"] for player in players]
    
longest_match = []
for game in timePerAction:
    for time in game:
        if len(time) > len(longest_match):
            longest_match = time
            
x_time = np.arange(0, len(longest_match))
y_time = np.array(longest_match)

# Set the y-axis limit starting from the minimum value
plt.title("Time per action") 
plt.xlabel("Action") 
plt.ylabel("Time in seconds")
plt.xlim(0, max(x_time))
plt.xticks(x_time)
plt.plot(x_time, y_time) 

# Create the plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

plt.savefig(f'plots/TimePerActioFig.svg', format='svg')
plt.show()
plt.close()