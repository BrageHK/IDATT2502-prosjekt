import pickle
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

def value_loss(folder):
    print("data/"+folder+"/loss_values-10.pk1")
    with open("data/"+folder+"/loss_values-10.pk1", "rb") as file:
        print("Inside")
        value_loss = pickle.load(file)
        
        x_value = np.arange(1, len(value_loss)+1)
        y_value = np.array(value_loss)
        
        plt.plot(x_value, y_value, label="Value loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss values")
        plt.legend()
        plt.show()

def loss(folder):
    print("data/"+folder+"/loss_values-10.pk1")
    with open("data/"+folder+"/loss_values-10.pk1", "rb") as file:
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
value_loss(folder)

