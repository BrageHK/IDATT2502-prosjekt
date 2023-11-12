import pickle
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

with open("data/test/loss_values.pk1", "rb") as file:
    policy_loss, value_loss = pickle.load(file)
    
    print("Ploicy loss: ", len(policy_loss))
    
    x_policy = np.arange(1, len(policy_loss) + 1) 
    y_policy = np.array(policy_loss)
    x_value = np.arange(1, len(value_loss) + 1) 
    y_value = np.array(value_loss)
    
    plt.title("Policy loss and value loss") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.plot(x_policy, y_policy, label = "Policy loss") 
    plt.plot(x_value, y_value, label = "Value loss") 
    plt.legend()
    plt.show()