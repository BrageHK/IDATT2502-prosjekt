from collections import deque
import numpy as np
import pickle


queue = deque(maxlen=200)
for i in range(300):
    queue.append(i)
    
print(np.random.choice(queue, 3))

with open("data/test/games.pk1", "rb") as file:
    positions = pickle.load(file)
    
print(len(positions))
print(positions)