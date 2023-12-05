import numpy as np

class Maze:
    def __init__(self, size = 1):
        self.shape = size * 2 + 3
        box = "\u25A2"
        end = "\u229A"

        self.state = np.full([self.shape, self.shape], ' ')
        self.state[0] = self.state[-1] = self.state[:,0] = self.state[:,-1] = np.full([self.shape], box)
        self.state[2:-2:2,2:-2:2] = np.full([size, size], box)
        self.state[1][1] = self.state[-2][-2] = end

        self.returns = np.zeros([self.shape, self.shape, 4])
        self.values = np.zeros([self.shape, self.shape])
        self.evaluate()
            
    def print(self, pi = None):
        stringify = self.state.copy()
        table = {   "[1.0, 0.0, 0.0, 0.0]": "\u2191", "[0.0, 1.0, 0.0, 0.0]": "\u2192", "[0.0, 0.0, 1.0, 0.0]": "\u2193", "[0.0, 0.0, 0.0, 1.0]": "\u2190", 
                    "[0.5, 0.0, 0.5, 0.0]": "\u2195", "[0.0, 0.5, 0.0, 0.5]": "\u2194", "[0.25, 0.25, 0.25, 0.25]": "\u254B", 
                    "[0.5, 0.5, 0.0, 0.0]": "\u2517", "[0.0, 0.5, 0.5, 0.0]": "\u250F", "[0.0, 0.0, 0.5, 0.5]": "\u2513", "[0.5, 0.0, 0.0, 0.5]": "\u251B", 
                    "[0.0, 0.33, 0.33, 0.33]": "\u2533", "[0.33, 0.0, 0.33, 0.33]": "\u252B", "[0.33, 0.33, 0.0, 0.33]": "\u253B", "[0.33, 0.33, 0.33, 0.0]": "\u2523" }
        pi = pi.round(2) if type(pi) == np.ndarray else np.full([self.shape, self.shape, 4], 0.25)
        for y in range(1, self.shape - 1):
            for x in range(1, self.shape - 1):
                if stringify[y][x] == ' ':
                    stringify[y][x] = table[str(pi[y][x].tolist())]
        print(np.array2string(stringify, separator = '', formatter = {'str_kind': lambda x: x}))

    def evaluate(self):
        self.rewards = np.zeros([self.shape, self.shape])
        for y in range(self.shape):
            for x in range(self.shape):
                if self.state[y][x] == "\u25A2":
                    self.rewards[y][x] = -10**3
                elif self.state[y][x] == "\u229A":
                    self.rewards[y][x] = 10
                elif self.state[y][x] == " ":
                    self.rewards = -1

    def scrabble(self, num = None):
        if num == None:
            num = (self.shape - 2)**2 // 10
        while num > 0:
            x, y = np.random.randint(1, self.shape - 1, 2)
            if self.state[y][x] == ' ':
                self.state[y][x] = "\u25A2"
                num -= 1
        self.evaluate()
        
# dynamic programming
# full backups
# class Maze:
#     def __init__(self, size = 1):
#         self.shape = size * 2 + 3
#         box = "\u25A2"
#         end = "\u229A"

#         self.state = np.full([self.shape, self.shape], ' ')
#         self.state[0] = self.state[-1] = self.state[:,0] = self.state[:,-1] = np.full([self.shape], box)
#         self.state[2:-2:2,2:-2:2] = np.full([size, size], box)
#         self.state[1][1] = self.state[-2][-2] = end

#         self.evaluate()
            
#     def print(self, pi = None):
#         stringify = self.state.copy()
#         table = {   "[1.0, 0.0, 0.0, 0.0]": "\u2191", "[0.0, 1.0, 0.0, 0.0]": "\u2192", "[0.0, 0.0, 1.0, 0.0]": "\u2193", "[0.0, 0.0, 0.0, 1.0]": "\u2190", 
#                     "[0.5, 0.0, 0.5, 0.0]": "\u2195", "[0.0, 0.5, 0.0, 0.5]": "\u2194", "[0.25, 0.25, 0.25, 0.25]": "\u254B", 
#                     "[0.5, 0.5, 0.0, 0.0]": "\u2517", "[0.0, 0.5, 0.5, 0.0]": "\u250F", "[0.0, 0.0, 0.5, 0.5]": "\u2513", "[0.5, 0.0, 0.0, 0.5]": "\u251B", 
#                     "[0.0, 0.33, 0.33, 0.33]": "\u2533", "[0.33, 0.0, 0.33, 0.33]": "\u252B", "[0.33, 0.33, 0.0, 0.33]": "\u253B", "[0.33, 0.33, 0.33, 0.0]": "\u2523" }
#         pi = pi.round(2) if type(pi) == np.ndarray else np.full([self.shape, self.shape, 4], 0.25)
#         for y in range(1, self.shape - 1):
#             for x in range(1, self.shape - 1):
#                 if stringify[y][x] == ' ':
#                     stringify[y][x] = table[str(pi[y][x].tolist())]
#         print(np.array2string(stringify, separator = '', formatter = {'str_kind': lambda x: x}))

#     def evaluate(self):
#         self.valuesNew = np.zeros([self.shape, self.shape])
#         for y in range(self.shape):
#             for x in range(self.shape):
#                 if self.state[y][x] == "\u25A2":
#                     self.valuesNew[y][x] = -10**3

#     def scrabble(self, num = None):
#         if num == None:
#             num = (self.shape - 2)**2 // 10
#         while num > 0:
#             x, y = np.random.randint(1, self.shape - 1, 2)
#             if self.state[y][x] == ' ':
#                 self.state[y][x] = "\u25A2"
#                 num -= 1
#         self.evaluate()