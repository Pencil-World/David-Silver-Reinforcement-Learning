import numpy as np
import random
import sys

# model-free prediction: 
# model-free control: 
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

maze = Maze(10)
maze.scrabble()
pi = np.full([maze.shape, maze.shape, 4], 0.25) # the policy chooses actions from north, east, south, west
epochs = 100

def policy_improvement():
    for y in range(1, maze.shape - 1):
        for x in range(1, maze.shape - 1):
            adjacent = maze.valuesOld[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = np.where(adjacent == adjacent.max())[0]
            pi[y][x] = np.zeros(4)
            pi[y][x][actions] = 1 / len(actions)

# dynamic programming
# k = 1 # how often to update the policy from the values. value iteration: k = 1, policy iteration: k > 1
# isSynchronous = True
# for i in range(epochs):
#     maze.valuesOld = maze.valuesNew.copy()
#     if i % k == 0:
#         policy_improvement()
#     for y in range(1, maze.shape - 1):
#         for x in range(1, maze.shape - 1):
#             if not isSynchronous:
#                 y, x = np.random.randint(1, maze.shape - 1, 2)
#             if maze.state[y][x] == ' ':
#                 actions = pi[y][x]
#                 maze.valuesNew[y][x] = sum(actions * maze.valuesOld[[y-1, y, y+1, y], [x, x+1, x, x-1]]) - 1

# what do i call this
step = 1 # number of steps to propogate the error. monte-carlo: step = infinity, td: step = 1
alpha = 0.001 # learning rate. how long the algorithm will remember past episodes
y = x = 0
for i in range(epochs):
    policy_improvement()
    while maze.state[y][x] != ' ':
        y, x = np.random.randint(1, maze.shape - 1, 2)

    history = []
    while maze.state[y][x] != "\u229A":
        adjacent = maze.valuesNew[[y-1, y, y+1, y], [x, x+1, x, x-1]]
        actions = np.where(adjacent == adjacent.max())[0]
        pi[y][x] = np.zeros(4)
        pi[y][x][actions] = 1 / len(actions)

        action = actions[random.randint(len(actions))]
        y += 0 if i % 2 == 0 else i - 1
        x += 0 if i % 2 == 1 else 2 - i
        history.append(i)

        if len(history) == step or len(history) < step and maze.state[y][x] == "\u229A":
            reward = maze.valuesNew[y][x]
            for y, x in history: # double check the order of the reward and the update
                reward += -1
                maze.valuesNew[y][x] = maze.valuesNew[y][x] + alpha * (reward - maze.valuesNew[y][x])
        else:
            y, x = history[-step]
            reward = maze.valuesNew[y][x] - step
            maze.valuesNew[y][x] = maze.valuesNew[y][x] + alpha * (reward - maze.valuesNew[y][x])

policy_improvement() # only on specific or episode data. this will be model free control. 

# td(lambda)
lamb = 1 # number of steps to propogate the error. monte-carlo: step = infinity, td: step = 1
alpha = 0.001 # learning rate. how long the algorithm will remember past episodes
y = x = 0

maze.print(pi)