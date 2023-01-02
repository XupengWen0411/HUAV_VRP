import math,random,time
import matplotlib.pyplot as plt


#Read coordinate file
filename = 'berlin52.txt'
city_num = []
city_location = []
with open(filename, 'r') as f:
    datas = f.readlines()[0:-1]
for data in datas:
    data = data.split()
    city_num.append(int(data[0]))
    x = float(data[1])
    y = float(data[2])
    city_location.append((x,y))

city_count = len(city_num)
origin = 1                    # Set starting node number
city_num.remove(origin)
remain_cities = city_num[:]   # Cities that change during iteration
remain_count = city_count - 1 # Number of cities changed during iteration

# Calculate adjacency matrix
dis =[[0]*city_count for i in range(city_count)] # initialization
for i in range(city_count):
    for j in range(city_count):
        if i != j:
            dis[i][j] = math.sqrt((city_location[i][0]-city_location[j][0])**2 + (city_location[i][1]-city_location[j][1])**2)
        else:
            dis[i][j] = 0

def route_mile_cost(route):
    '''
    Calculate distance cost
    '''
    mile_cost = 0.0
    mile_cost += dis[origin-1][route[0]-1]
    for i in range(remain_count-1):
        mile_cost += dis[route[i]-1][route[i+1]-1]
    mile_cost += dis[route[-1]-1][origin-1]
    return mile_cost

# Get the shortest city in the current neighbor city
def nearest_city(current_city,remain_cities):
    temp_min = float('inf')
    next_city = None
    for i in range(len(remain_cities)):
        distance = dis[current_city-1][remain_cities[i]-1]
        if distance < temp_min:
            temp_min = distance
            next_city = remain_cities[i]
    return next_city,temp_min

def greedy_initial_route(remain_cities):
    '''
    Using greedy algorithm to generate initial solution
    '''
    cand_cities = remain_cities[:]
    current_city = origin
    initial_route = []
    mile_cost = 0
    while len(cand_cities) > 0:
        next_city,distance = nearest_city(current_city,cand_cities)
        mile_cost += distance
        initial_route.append(next_city)
        current_city = next_city
        cand_cities.remove(next_city)
    mile_cost += dis[initial_route[-1]-1][origin-1]
    return initial_route,mile_cost

indexs = list(k for k in range(remain_count))
def random_swap_2_city(route):
    '''
    2-opt swap operator
    '''
    new_route = route[:]
    index = sorted(random.sample(indexs,2))
    L = index[1] - index[0] + 1
    for j in range(L):
        new_route[index[0]+j] = route[index[1]-j]
    return new_route

def main():
    T=3000;Tfloor=1;alpha=0.995;iter_count=100
    route = city_num[:]
    random.shuffle(route)
    mile = route_mile_cost(route)
    # route,mile = greedy_initial_route(remain_cities)
    best_route,best_value = route[:],mile
    record = [best_value]
    while T > Tfloor:
        for i in range(iter_count):
            cand_route = random_swap_2_city(route)
            cand_mile = route_mile_cost(cand_route)
            # If the candidate solution is better than the current optimal solution,
            # update the current optimal solution
            if cand_mile <= mile:
                route = cand_route[:]
                mile = cand_mile
                best_value = mile
                best_route = route
                # T -= mile - cand_mile
            # If the candidate solution is worse than the current optimal solution,
            # accept the candidate solution according to probability
            else:
                p = math.exp((mile - cand_mile)/T)
                if random.random() < p:
                    route = cand_route[:]
                    mile = cand_mile
        T *= alpha # Simulated annealing cooling
        record.append(best_value)
    best_route = [origin] + best_route + [origin]
    return best_route,best_value,record

def fig():
    time_start = time.time()
    satisfactory_solution,mile_cost,record = main()
    time_end = time.time()
    time_cost = time_end - time_start
    print('time cost:',time_cost)
    print("Costs:%d" %(int(mile_cost)))
    print("Routes:\n",satisfactory_solution)

    X = []
    Y = []
    for i in satisfactory_solution:
        x = city_location[i-1][0]
        y = city_location[i-1][1]
        X.append(x)
        Y.append(y)
    plt.scatter(x,y)
    plt.plot(X,Y,'-o')
    plt.title("satisfactory solution of TS:%d"%(int(mile_cost)))
    plt.show()
    plt.xlabel('temperature',fontproperties="SimSun")
    plt.ylabel('Route length',fontproperties="SimSun")
    plt.title("Solution of SA changed with temperature")
    plt.plot(record,'-')
    plt.show()
    return mile_cost,time_cost

fig()

