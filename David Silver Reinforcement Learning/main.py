"""
There is a grid. 
There are evacuation zones throughout the grid. 
there are walls/obstacles throughout the grid. 
Given any position on the grid, what is the most efficient way to traverse to an evacuation zone?
"""

from collections import deque
from Maze import Maze
import numpy as np
import random
#from tensorflow import keras
#import xgboost # much preferred

# model-free prediction: estimate the value function of an unknown MDP
# model-free control: optimise the value function of an unknown MDP
# bootstrapping: update involves an estimate
# sampling: update samples an expectation
# model-free: no advance knowledge of MDP required
# model-based: uses the model's predictions or distributions of next state and reward in order to calculate optimal actions like dynamic programming
# on-policy learning: "learning on the job," learn about policy pi from experience sampled form pi
# off-policy learning: "look over someone's shoulder," learn about policy pi from experience sampled from mu
# online learning: updates are applied at each step within the episode and doesn't require non-terminating environments
# offline learning: updates are accumulated within episode but applied in batch at the end of episode
# synchronous backups: all states are backed up in parallel. always converges
# asynchronous backups: backs up states individually in any order. converges when all states are selected
# all algorithms will be undiscounted so gamma is assumed to be one. rewards will not vary with action. environments will not be stochastic. 

maze = Maze(3)
maze.scrabble()
pi = np.full([maze.shape, maze.shape, 4], 0.25) # the policy chooses actions from north, east, south, west
epochs = 100
maze.print()

# def policy_evaluation(y, x, epsilon):
#     if random.randint(100) < epsilon:
#         adjacent = maze.returns[y][x]
#         actions = np.where(adjacent == adjacent.max())[0]
#     else:
#         adjacent = maze.rewards[[y-1, y, y+1, y], [x, x+1, x, x-1]]
#         actions = [elem for elem in adjacent if elem > -10**3]
#     return actions[random.randint(len(actions))]

# def policy_improvement(y, x):
#     adjacent = maze.returns[y, x]
#     actions = np.where(adjacent == adjacent.max())[0]
#     pi[y][x] = np.zeros(4)
#     pi[y][x][actions] = 1 / len(actions)
        
def policy_evaluation(y, x, epsilon):
    if random.random() < epsilon:
        adjacent = maze.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]
        actions = np.where(adjacent == adjacent.max())[0]
    else:
        adjacent = maze.rewards[[y-1, y, y+1, y], [x, x+1, x, x-1]]
        actions = [i for i, elem in enumerate(adjacent) if elem > -10**3]
    return actions[random.randint(0, len(actions) - 1)]

def policy_improvement(y, x):
    adjacent = maze.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]
    actions = np.where(adjacent == adjacent.max())[0]
    pi[y][x] = np.zeros(4)
    pi[y][x][actions] = 1 / len(actions)
    
# dynamic programming
# k = 1 # how often to update the policy from the values. value iteration: k = 1, policy iteration: k > 1
# isSynchronous = True
# for i in range(epochs):
#    maze.valuesOld = maze.valuesNew.copy()
#    if i % k == 0:
#         for y in range(1, maze.shape - 1):
#            for x in range(1, maze.shape - 1):
#                policy_improvement(y, x)
#    for y in range(1, maze.shape - 1):
#        for x in range(1, maze.shape - 1):
#            if not isSynchronous:
#                y, x = np.random.randint(1, maze.shape - 1, 2)
#            if maze.state[y][x] == ' ':
#                actions = pi[y][x]
#                maze.valuesNew[y][x] = sum(actions * maze.valuesOld[[y-1, y, y+1, y], [x, x+1, x, x-1]]) - 1

# td(lambda)
step = 1 # number of steps to propogate the error. monte-carlo: step = infinity, td: step = 1
alpha = 0.001 # learning rate. how well the algorithm remembers past episodes
gamma = 0.99
y = x = 0
for i in range(epochs):
    # initialize the episode
    while maze.state[y][x] != ' ':
        y, x = np.random.randint(1, maze.shape - 1, 2)

    history = deque()
    epsilon = i / epochs
    while (debug := maze.state[y][x]) != "\u229A":
        print((y, x))
        if debug == '\u25A2':
            print("Oops! Looks like there's an error. ")
        history.append((y, x))
        if len(history) - 1 == step:
            reward = 0
            for y, x in list(history)[::-1]:
                reward = maze.rewards[y][x] + gamma * reward
            maze.values[y][x] = maze.values[y][x] + alpha * (reward - maze.values[y][x])
            y, x = history[-1]
            history.popleft()

        action = policy_evaluation(y, x, epsilon)
        y += 0 if action % 2 == 1 else action - 1
        x += 0 if action % 2 == 0 else 2 - action
    reward = maze.values[y][x]
    for y, x in list(history)[::-1]:
        reward = maze.rewards[y][x] + gamma * reward
        maze.values[y][x] = maze.values[y][x] + alpha * (reward - maze.values[y][x])
for y in range(len(maze.values)):
    for x in range(len(maze.values[y])):
        policy_improvement(y, x)

# next learning mumbo jumbo

maze.print(pi)