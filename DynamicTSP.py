
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  


# Using dynamic programming method to solve TSP
class DynaticTSP:
    def __init__(self,cities, X, start_node):
        self.cities = cities
        self.X = X               # Distance matrix
        self.start_node = start_node
        self.array = [[0] * (2 ** (len(self.X) - 1)) for i in range(len(self.X))]

    def transfer(self, sets):
        su = 0
        for s in sets:
            su = su + 2 ** (s - 1)
        return su

    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num))
        cities.pop(cities.index(s))
        node = s
        return self.solve(node, cities)

    def solve(self, node, future_sets):
        if len(future_sets) == 0:
            return self.X[node][self.start_node]
        d = 99999
        distance = []
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i)
            distance.append(self.X[node][s_i] + self.solve(s_i, copy))
        # Recursive equation of dynamic programming
        d = min(distance)
        next_one = future_sets[distance.index(d)]
        # Node set not traversed
        c = self.transfer(future_sets)
        # Backtracking matrix
        self.array[node][c] = next_one
        return d
