import pickle
from collections import deque

with open("data/test/loss_values.pk1", "rb") as file:
    policy_loss, value_loss = pickle.load(file)
    
    
    
    print("Policy loss data poins:")
    print(len(policy_loss))
    print("first loss value: ", policy_loss)
    # print("Value loss data points:")
    # print(len(value_loss))