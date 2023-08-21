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

def policy_improvement(y, x):
        adjacent = maze.valuesOld[[y-1, y, y+1, y], [x, x+1, x, x-1]]
        actions = np.where(adjacent == adjacent.max())[0]
        pi[y][x] = np.zeros(4)
        pi[y][x][actions] = 1 / len(actions)

# dynamic programming
#k = 1 # how often to update the policy from the values. value iteration: k = 1, policy iteration: k > 1
#isSynchronous = True
#for i in range(epochs):
#    maze.valuesOld = maze.valuesNew.copy()
#    if i % k == 0:
        #for y in range(1, maze.shape - 1):
        #    for x in range(1, maze.shape - 1):
        #        policy_improvement(y, x)
#    for y in range(1, maze.shape - 1):
#        for x in range(1, maze.shape - 1):
#            if not isSynchronous:
#                y, x = np.random.randint(1, maze.shape - 1, 2)
#            if maze.state[y][x] == ' ':
#                actions = pi[y][x]
#                maze.valuesNew[y][x] = sum(actions * maze.valuesOld[[y-1, y, y+1, y], [x, x+1, x, x-1]]) - 1

# forward view eligibility traces
step = 1 # number of steps to propogate the error. monte-carlo: step = infinity, td: step = 1
alpha = 0.001 # learning rate. how well the algorithm remembers past episodes
epsilon = 95
gamma = 0.99
y = x = 0
maze.valuesOld = maze.valuesNew.copy
rewards = np.full([maze.shape, maze.shape], -1)
rewards[1][1] = rewards[-2][-2] = 10
for i in range(epochs):
    # initialize the episode
    policy_improvement()
    while maze.state[y][x] != ' ':
        y, x = np.random.randint(1, maze.shape - 1, 2)

    history = deque()
    while maze.state[y][x] != "\u229A":
        if random.randint(100) < epsilon:
            adjacent = maze.valuesNew[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = np.where(adjacent == adjacent.max())[0]
            policy_improvement(y, x)
            action = actions[random.randint(len(actions))]
        else:
            print()

        y += 0 if action % 2 == 0 else action - 1
        x += 0 if action % 2 == 1 else 2 - action
        history.append(reward)
        if len(history) == step or (len(history) < step and maze.state[y][x] == "\u229A"):
            reward = maze.valuesNew[y][x]
            for y, x in history:
                reward += -1
                maze.valuesNew[y][x] = maze.valuesNew[y][x] + alpha * (reward - maze.valuesNew[y][x])
        else:
            y, x = history[-step]
            reward = maze.valuesNew[y][x] - step
            maze.valuesNew[y][x][action] = maze.valuesNew[y][x] + alpha * (reward - maze.valuesNew[y][x])

## td(lambda)
#lamb = 1 # number of steps to propogate the error. monte-carlo: step = infinity, td: step = 1
#alpha = 0.001 # learning rate. how long the algorithm will remember past episodes
#y = x = 0

# next learning mumbo jumbo

maze.print(pi)