import pickle
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os

def game_length(folder, batch_size):
    with open("data/"+folder+"/game_lengths/game_lengths.pt", "rb") as file:
        game_length = pickle.load(file)
        
        y_value = np.array(game_length)
        new_y_values = []
        
        counter = 0
        sum = 0
        for i in range(0, len(y_value)):
            if counter == batch_size:
                new_y_values.append(sum/batch_size)
                sum = 0
                counter = 0
            else:
                sum += y_value[i]
            counter = counter + 1
        
        
        plt.axhline(y = 42, color = 'r', linestyle = 'dashed', label="42 moves") 
        plt.plot(np.arange(1, len(new_y_values)+1), np.array(new_y_values), label="Game length")
        plt.xlabel("Iterations")
        plt.ylabel("Number of moves")
        plt.title("Game length")
        plt.legend()
        
        # Create the plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/game_length.svg', format='svg')
        plt.show()

def value_loss(folder):
    print("data/"+folder+"/loss_values/loss_values.pt")
    with open("data/"+folder+"/loss_values/loss_values.pt", "rb") as file:
        print("Inside")
        value_loss = pickle.load(file)
        
        x_value = np.arange(1, len(value_loss)+1)
        y_value = np.array(value_loss)
        
        plt.plot(x_value, y_value, label="Value loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss values")
        plt.legend()
        
        # Create the plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/value_loss.svg', format='svg')
        plt.show()

def loss(folder):
    print("data/"+folder+"/loss_values/loss_values.pt")
    with open("data/"+folder+"/loss_values/loss_values.pt", "rb") as file:
        print("Inside")
        policy_loss, value_loss = pickle.load(file)
        
        print("Policy loss data poins:")
        # print(len(policy_loss))
        # for i in range(10):
        #     print(policy_loss[i])
        #print("first loss value: ", policy_loss)
        # print("Value loss data points:")
        # print(len(value_loss))
        
        x_policy = np.arange(1, len(policy_loss)+1)
        y_policy = np.array(policy_loss)
        x_value = np.arange(1, len(value_loss)+1)
        y_value = np.array(value_loss)
        
        
        plt.plot(x_policy, y_policy, label="Policy loss")
        plt.plot(x_value, y_value, label="Value loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss values")
        plt.legend()
        
        # Create the plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')

        # Save the plot as SVG
        plt.savefig(f'plots/loss.svg', format='svg')
        plt.show()
        
def games(folder):
    with open("data/"+folder+"/games.pk1", "rb") as file:
        positions = pickle.load(file)
        print(len(positions))
        # First 10 games:
        for i in range(10):
            print(positions[i])

def create_dummy_data():
    with open("data/TicTacToe/lol.pk1", "wb") as file:
        pickle.dump((np.array([1,2,3]), np.array([4,5,6])), file)


folder = "ConnectFour"
#games(folder)
game_length(folder, 500)

