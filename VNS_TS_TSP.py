import math,random,time
import matplotlib.pyplot as plt

class VNS_TS:
    def __init__(self, city_location):
        self.city_location = city_location
        self.city_count = len(self.city_location)
        self.city_num = list(range(0, self.city_count))
        self.dis = self.distance()
        self.origin = 1
        self.remain_cities = self.city_num[:]
        self.remain_cities.remove(self.origin)
        self.remain_count = self.city_count - 1
        self.improve_count = 100
        self.candiate_routes_size = 1.5*self.city_count
        self.tabu_size = 10
        self.tabu_list = []
        self.best_so_far_cost = 0
        self.best_so_far_route = []
        self.tabu_list = []

    # Calculate distance adjacency matrix
    def distance(self):
        dis =[[0]*self.city_count for i in range(self.city_count)]
        for i in range(self.city_count):
            for j in range(self.city_count):
                if i != j:
                    dis[i][j] = math.sqrt((self.city_location[i][0]-self.city_location[j][0])**2 + (self.city_location[i][1]-self.city_location[j][1])**2)
                else:
                    dis[i][j] = 0
        return dis

    def route_mile_cost(self,route):
        '''
        Calculate distance cost
        '''
        mile_cost = 0.0
        mile_cost += self.dis[self.origin-1][route[0]-1]
        for i in range(self.remain_count-1):
            mile_cost += self.dis[route[i]-1][route[i+1]-1]
        mile_cost += self.dis[route[-1]-1][self.origin-1]
        return mile_cost

    def random_initial_route(self,remain_cities):
        '''
        Generate Initial Route
        '''
        initial_route = remain_cities[:]
        random.shuffle(initial_route)
        mile_cost = self.route_mile_cost(initial_route)
        return initial_route,mile_cost

    def improve_circle(self,remain_cities):
        '''
        Improved algorithm to generate initial route
        '''
        initial_route = remain_cities[:]
        random.shuffle(initial_route)
        cost0 = self.route_mile_cost(initial_route)
        route = [1] + initial_route + [1]
        label = list(i for i in range(1,len(remain_cities)))
        j = 0
        while j < self.improve_count:
            new_route = route[:]
            index0,index1 = random.sample(label,2)
            new_route[index0],new_route[index1]= new_route[index1],new_route[index0]
            cost1 = self.route_mile_cost(new_route[1:-1])
            improve = cost1 - cost0
            if improve < 0: #交Improvements after exchanging two nodes
                route = new_route[:]
                cost0 = cost1
                j += 1
            else:
                continue
        initial_route = route[1:-1]
        return initial_route,cost0

    # Get the shortest city in the current neighbor city
    def nearest_city(self,current_city,cand_cities):
        temp_min = float('inf')
        next_city = None
        for i in range(len(cand_cities)):
            distance = self.dis[current_city-1][cand_cities[i]-1]
            if distance < temp_min:
                temp_min = distance
                next_city = cand_cities[i]
        return next_city,temp_min

    def greedy_initial_route(self,remain_cities):
        '''
        Using greedy algorithm to generate initial solution
        '''
        cand_cities = remain_cities[:]
        current_city = self.origin
        mile_cost = 0
        initial_route = []
        while len(cand_cities) > 0:
            next_city,distance = self.nearest_city(current_city,cand_cities)
            mile_cost += distance
            initial_route.append(next_city)
            current_city = next_city
            cand_cities.remove(next_city)
        mile_cost += self.dis[initial_route[-1]-1][0]
        return initial_route,mile_cost

    def random_swap_2_city(self,route):
        '''
        2-opt swap operator
        '''
        new_route = route[:]
        swap_2_city = random.sample(route,2)
        index = [0]*2
        index[0] = route.index(swap_2_city[0])
        index[1] = route.index(swap_2_city[1])
        index = sorted(index)
        L = index[1] - index[0] + 1
        for j in range(L):
            new_route[index[0]+j] = route[index[1]-j]
        return new_route,sorted(swap_2_city)

    def generate_new_route(self,route):
        '''
        Generate new route
        '''
        candidate_routes = []
        candidate_mile_cost = []
        candidate_swap = []
        while len(candidate_routes) < self.candiate_routes_size:
            cand_route,cand_swap = self.random_swap_2_city(route)
            if cand_swap not in candidate_swap:
                candidate_routes.append(cand_route)
                candidate_swap.append(cand_swap)
                candidate_mile_cost.append(self.route_mile_cost(cand_route))
        min_mile_cost = min(candidate_mile_cost)
        i = candidate_mile_cost.index(min_mile_cost)
        # If the optimal value of this exchange set is better than the historical optimal value,
        # update the historical optimal value and the optimal route
        if min_mile_cost < self.best_so_far_cost:
            self.best_so_far_cost = min_mile_cost
            self.best_so_far_route = candidate_routes[i]
            new_route = candidate_routes[i]
            if candidate_swap[i] in self.tabu_list:
                self.tabu_list.remove(candidate_swap[i]) # Flout law
            elif len(self.tabu_list) >= self.tabu_size:
                self.tabu_list.remove(self.tabu_list[0])
            self.tabu_list.append(candidate_swap[i])
        else:
            # If no better route is found for this exchange set,
            # the sub optimal exchange method not in the tabu list will be selected
            K = self.candiate_routes_size
            stop_value = K - len(self.tabu_list) - 1
            while K > stop_value:
                min_mile_cost = min(candidate_mile_cost)
                i = candidate_mile_cost.index(min_mile_cost)
                # If the optimal value of this exchange set is better than the historical optimal value,
                # update the historical optimal value and the optimal route
                if min_mile_cost < self.best_so_far_cost:
                    self.best_so_far_cost = min_mile_cost
                    self.best_so_far_route = candidate_routes[i]
                    new_route = candidate_routes[i]
                    if candidate_swap[i] in self.tabu_list:
                        self.tabu_list.remove(candidate_swap[i]) # Flout law
                    elif len(self.tabu_list) >= self.tabu_size:
                        self.tabu_list.remove(self.tabu_list[0])
                    self.tabu_list.append(candidate_swap[i])
                    break
                else:
                    # If no better route is found for this exchange set,
                    # the sub optimal exchange method not in the tabu list will be selected
                    if candidate_swap[i] not in self.tabu_list:
                        self.tabu_list.append(candidate_swap[i])
                        new_route = candidate_routes[i]
                        if len(self.tabu_list) > self.tabu_size:
                            self.tabu_list.remove(self.tabu_list[0])
                        break
                    else:
                        candidate_mile_cost.remove(min_mile_cost)
                        candidate_swap.remove(candidate_swap[i])
                        candidate_routes.remove(candidate_routes[i])
                        K -= 1
        return new_route

    def tabu_search(self,remain_cities,iteration_count=400):
        self.best_so_far_route,self.best_so_far_cost =self.greedy_initial_route(remain_cities)
        # best_so_far_route,best_so_far_cost = random_initial_route(remain_cities)
        # best_so_far_route,best_so_far_cost = improve_circle(remain_cities)
        record = [self.best_so_far_cost]
        new_route = self.best_so_far_route[:]
        for j in range(iteration_count):
            new_route = self.generate_new_route(new_route)
            record.append(self.best_so_far_cost)
        final_route = [self.origin] + self.best_so_far_route +[self.origin]
        return final_route,self.best_so_far_cost,record

    def plot(self, N, mile_cost, record, satisfactory_solution, city_location):
        X = []
        Y = []
        for i in satisfactory_solution:
            x = city_location[i - 1][0]
            y = city_location[i - 1][1]
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, '-o')
        plt.title("Large drone routes:%d" % (int(mile_cost)))
        plt.show()

        A = [i for i in range(N + 1)]
        B = record[:]
        plt.xlim(0, N)
        plt.xlabel('Iterations', fontproperties="SimSun")
        plt.ylabel('Route distance', fontproperties="SimSun")
        plt.title("Solution of TS changed with iteration")
        plt.plot(A, B, '-')
        plt.show()

def loadData():
    filename = 'Data\Random\data50.txt'
    city_location = []
    with open(filename, 'r') as f:
        datas = f.readlines()
    for data in datas:
        data = data.split()
        x = float(data[0])
        y = float(data[1])
        city_location.append((x, y))
    return city_location

if __name__ == '__main__':
    N = 100         # The number of iterations
    time_start = time.time()
    city_location = loadData()
    vns_ts = VNS_TS(city_location)
    satisfactory_solution, mile_cost, record = vns_ts.tabu_search(vns_ts.remain_cities, N)
    time_end = time.time()
    time_cost = time_end - time_start
    print('time cost:', time_cost)
    print("Costs:%d" %(int(mile_cost)))
    print("Optimization Routes:\n", satisfactory_solution)
    vns_ts.plot(N, mile_cost, record, satisfactory_solution, vns_ts.city_location)
