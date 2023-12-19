import numpy as np
import random
goal, space, obstacle = 10, -1, -10**3 # rename
space, obstacle, goal = 0, 0, 0 # given a space, manuver around obstacles to reach a goal

class Maze:
    def __init__(self, size = 1):
        self.shape = size * 2 + 3
        self.values = np.zeros([self.shape, self.shape])
            
        self.rewards = np.full([self.shape, self.shape], space) # full or fill
        self.rewards[0] = self.rewards[-1] = self.rewards[:,0] = self.rewards[:,-1] = np.full([self.shape], obstacle)
        self.rewards[2:-2:2,2:-2:2] = np.full([size, size], obstacle)
        self.rewards[1][1] = self.rewards[-2][-2] = goal

        self.pi = np.zeros([self.shape, self.shape, 4])
        self.returns = np.zeros([self.shape, self.shape, 4])

    def __str__(self):
        box = "\u25A2"
        end = "\u229A"
        self.state = self.rewards.copy()
        table = {   "[1.0, 0.0, 0.0, 0.0]": "\u2191", "[0.0, 1.0, 0.0, 0.0]": "\u2192", "[0.0, 0.0, 1.0, 0.0]": "\u2193", "[0.0, 0.0, 0.0, 1.0]": "\u2190", 
                    "[0.5, 0.0, 0.5, 0.0]": "\u2195", "[0.0, 0.5, 0.0, 0.5]": "\u2194", "[0.25, 0.25, 0.25, 0.25]": "\u254B", 
                    "[0.5, 0.5, 0.0, 0.0]": "\u2517", "[0.0, 0.5, 0.5, 0.0]": "\u250F", "[0.0, 0.0, 0.5, 0.5]": "\u2513", "[0.5, 0.0, 0.0, 0.5]": "\u251B", 
                    "[0.0, 0.33, 0.33, 0.33]": "\u2533", "[0.33, 0.0, 0.33, 0.33]": "\u252B", "[0.33, 0.33, 0.0, 0.33]": "\u253B", "[0.33, 0.33, 0.33, 0.0]": "\u2523" }
        self.pi = self.pi.round(2) if type(self.pi) == np.ndarray else np.full([self.shape, self.shape, 4], 0.25)
        for y in range(1, self.shape - 1):
            for x in range(1, self.shape - 1):
                if stringify[y][x] == ' ':
                    stringify[y][x] = table[str(self.pi[y][x].tolist())]
        print(np.array2string(stringify, separator = '', formatter = {'str_kind': lambda x: x}))

#-------------------------------------------

        stringify = self.state.copy()
        table = {   "[1.0, 0.0, 0.0, 0.0]": "\u2191", "[0.0, 1.0, 0.0, 0.0]": "\u2192", "[0.0, 0.0, 1.0, 0.0]": "\u2193", "[0.0, 0.0, 0.0, 1.0]": "\u2190", 
                    "[0.5, 0.0, 0.5, 0.0]": "\u2195", "[0.0, 0.5, 0.0, 0.5]": "\u2194", "[0.25, 0.25, 0.25, 0.25]": "\u254B", 
                    "[0.5, 0.5, 0.0, 0.0]": "\u2517", "[0.0, 0.5, 0.5, 0.0]": "\u250F", "[0.0, 0.0, 0.5, 0.5]": "\u2513", "[0.5, 0.0, 0.0, 0.5]": "\u251B", 
                    "[0.0, 0.33, 0.33, 0.33]": "\u2533", "[0.33, 0.0, 0.33, 0.33]": "\u252B", "[0.33, 0.33, 0.0, 0.33]": "\u253B", "[0.33, 0.33, 0.33, 0.0]": "\u2523" }
        self.pi = self.pi.round(2) if type(self.pi) == np.ndarray else np.full([self.shape, self.shape, 4], 0.25)
        for y in range(1, self.shape - 1):
            for x in range(1, self.shape - 1):
                if stringify[y][x] == ' ':
                    stringify[y][x] = table[str(self.pi[y][x].tolist())]
        print(np.array2string(stringify, separator = '', formatter = {'str_kind': lambda x: x}))

    # rename or remove?
    def evaluate(self):
        for y in range(self.shape):
            for x in range(self.shape):
                if self.state[y][x] == "\u25A2":
                    self.values[y][x] = self.rewards[y][x] = -10**3
                elif self.state[y][x] == "\u229A":
                    self.values[y][x] = self.rewards[y][x] = 10
                elif self.state[y][x] == " ":
                    self.rewards[y][x] = -1

    def scrabble(self, num = None):
        if num == None:
            num = (self.shape - 2)**2 // 10
        while num > 0:
            x, y = np.random.randint(1, self.shape - 1, 2)
            if self.state[y][x] == ' ':
                self.state[y][x] = "\u25A2"
                num -= 1
        self.evaluate()

    def evaluation(self):
        pass
    
    def improvement(self):
        pass
        
    def value_evaluation(self, y, x, epsilon):
        if random.random() < epsilon:
            adjacent = self.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = np.where(adjacent == adjacent.max())[0]
        else:
            adjacent = self.rewards[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = [i for i, elem in enumerate(adjacent) if elem > -10**3]
        return actions[random.randint(0, len(actions) - 1)]

    def value_improvement(self, y, x):
        adjacent = self.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]
        actions = np.where(adjacent == adjacent.max())[0]
        self.pi[y][x] = np.zeros(4)
        self.pi[y][x][actions] = 1 / len(actions)
        
    def return_evaluation(self, y, x, epsilon):
        if random.random() < epsilon:
            adjacent = self.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = np.where(adjacent == adjacent.max())[0]
        else:
            adjacent = self.rewards[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = [i for i, elem in enumerate(adjacent) if elem > -10**3]
        return actions[random.randint(0, len(actions) - 1)]

    def return_improvement(self, y, x):
        adjacent = self.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]
        actions = np.where(adjacent == adjacent.max())[0]
        self.pi[y][x] = np.zeros(4)
        self.pi[y][x][actions] = 1 / len(actions)