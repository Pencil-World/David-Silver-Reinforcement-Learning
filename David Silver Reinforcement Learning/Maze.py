import numpy as np
import random
space, obstacle, goal = -1, -10**3, 10 # given a space, manuver around obstacles to reach a goal

class Maze:
    def __init__(self, size = 1):
        self.shape = size * 2 + 3
        self.values = np.zeros([self.shape, self.shape])
        self.returns = np.zeros([self.shape, self.shape, 4])
            
        self.rewards = np.full([self.shape, self.shape], space) # full or fill
        self.rewards[0] = self.rewards[-1] = self.rewards[:,0] = self.rewards[:,-1] = np.full([self.shape], obstacle)
        self.rewards[2:-2:2,2:-2:2] = np.full([size, size], obstacle)
        self.rewards[1][1] = self.rewards[-2][-2] = goal

        func = lambda x: np.full([self.shape], (3*[0])[:x] + [obstacle] + (3*[0])[x:])
        self.pi = np.full([self.shape, self.shape, 4], 0.25)
        self.pi[1], self.pi[-2] = func(0), func(2)
        self.rewards[:,1], self.rewards[:,-2] = func(3), func(1)

    def __str__(self):
        self.state = self.rewards.copy()
        table = {   "[1.0, 0.0, 0.0, 0.0]": "\u2191", "[0.0, 1.0, 0.0, 0.0]": "\u2192", "[0.0, 0.0, 1.0, 0.0]": "\u2193", "[0.0, 0.0, 0.0, 1.0]": "\u2190", 
                    "[0.5, 0.0, 0.5, 0.0]": "\u2195", "[0.0, 0.5, 0.0, 0.5]": "\u2194", "[0.25, 0.25, 0.25, 0.25]": "\u254B", 
                    "[0.5, 0.5, 0.0, 0.0]": "\u2517", "[0.0, 0.5, 0.5, 0.0]": "\u250F", "[0.0, 0.0, 0.5, 0.5]": "\u2513", "[0.5, 0.0, 0.0, 0.5]": "\u251B", 
                    "[0.0, 0.33, 0.33, 0.33]": "\u2533", "[0.33, 0.0, 0.33, 0.33]": "\u252B", "[0.33, 0.33, 0.0, 0.33]": "\u253B", "[0.33, 0.33, 0.33, 0.0]": "\u2523", 
                    str(space): " ", str(obstacle): "\u25A2", str(goal): "\u229A" }
        
        for y in range(self.shape):
            for x in range(self.shape):
                self.state[y][x] = table[str(self.pi[y][x].round(2).tolist() if self.rewards[y][x] == space else self.rewards[y][x])]
        print(np.array2string(self.state, separator = '', formatter = {'str_kind': lambda x: x}))

    def scrabble(self, num = None):
        if num == None:
            num = (self.shape - 2)**2 // 10
        while num > 0:
            x, y = np.random.randint(1, self.shape - 1, 2)
            if self.state[y][x] == ' ':
                self.state[y][x] = "\u25A2"
                # change adjacent pi values
                num -= 1

    def evaluation(self, y, x, epsilon):
        if random.random() < epsilon:
            adjacent = self.pi[y][x]
            actions = np.where(adjacent == adjacent.max())[0]
        else:
            adjacent = self.rewards[[y-1, y, y+1, y], [x, x+1, x, x-1]]
            actions = [i for i, elem in enumerate(adjacent) if elem > obstacle]
        return actions[random.randrange(0, len(actions))]
    
    def improvement(self, y, x, func):
        adjacent = func(self, y, x)
        actions = np.where(adjacent == adjacent.max())[0]
        self.pi[y][x] = np.zeros(4)
        self.pi[y][x][actions] = 1 / len(actions)
    
    @staticmethod
    def _value():
        return lambda obj, y, x: obj.values[[y-1, y, y+1, y], [x, x+1, x, x-1]]

    @staticmethod
    def _return():
        return lambda obj, y, x: obj.returns[y][x]