import numpy as np
import matplotlib.pyplot as plt
import time
from Transform2xy import Transform2xy
from FCMCluster import FCM
from VNS_TS_TSP import VNS_TS
from DynamicTSP import DynaticTSP
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Dynamic programming algorithm
def dynamicTSPSolve(cities, distence_matrix):
    S = DynaticTSP(cities, distence_matrix, 0)
    print("Minimum distance:" + str(S.tsp()))
    # Start backtracking
    M = S.array
    lists = list(range(len(S.X)))
    start = S.start_node
    city_order = []
    while len(lists) > 0:
        lists.pop(lists.index(start))
        m = S.transfer(lists)
        next_node = S.array[start][m]
        print(start,"--->" ,next_node)
        city_order.append(cities[start])
        start = next_node
    x1 = []
    y1 = []
    for city in city_order:
        x1.append(city[0])
        y1.append(city[1])
    return x1,y1

# Routing the end small UAVs
def generateEndUAVRoute(FCMRes):
    Clast = FCMRes.Clast
    dict_cluster = FCMRes.dict_cluster
    airports = FCMRes.airports

    airportselect = []
    for cc in Clast:
        tempdis = 10000000
        index = -1
        for i in range(len(airports)):
            temp = (cc[0]-airports[i][0])**2 + (cc[1]-airports[i][1])**2
            if temp < tempdis:
                tempdis = temp
                index = i
        if index != -1:
            airportselect.append(index)

    endUAVRoutes = []
    for key in dict_cluster:
        cities = FCMRes.data[dict_cluster[key]]
        cities = np.insert(cities, 0, np.array([airports[airportselect[key]]]),0)
        cities = cities.reshape(len(dict_cluster[key]) + 1, 2)
        distence_matrix = FCMRes.distance(cities)
        xRoute, yRoute = dynamicTSPSolve(cities, distence_matrix)
        endUAVRoutes.append([key, [xRoute, yRoute]])
    return endUAVRoutes

# Plot the experiment results
def plot(FCMRes):
    mark = ['or', 'ob', 'og', 'om', 'oy', 'oc', 'sr', 'sb', 'sg', 'sm', 'sy', 'sc']
    # The first Pic (Scatter chart before clustering)
    plt.subplot(231)
    data = FCMRes.data
    plt.plot(data[:, 0], data[:, 1], 'ob', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot before clustering')

    # The second Pic(Results after clustering)
    plt.subplot(232)
    j = 0
    for type in FCMRes.label:
        plt.plot(FCMRes.data[j:j + 1, 0], FCMRes.data[j:j + 1, 1], mark[type % 12], markersize=2)
        j += 1
    plt.plot(FCMRes.Clast[:, 0], FCMRes.Clast[:, 1], 'k*', markersize=4)
    plt.plot(FCMRes.airports[:, 0], FCMRes.airports[:, 1], 'r*', markersize=6)
    disA = FCMRes.clusterCircle()
    ri = 0
    for key, v in disA:
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = v[1] + v[0] / 1.5 * np.cos(theta)
        y = v[2] + v[0] / 1.5 * np.sin(theta)
        ri = ri + 1
        plt.plot(x, y, mark[key % 12], markersize=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Clustering results")

    # The third Pic (Objective Function Value curves)
    plt.subplot(233)
    xxx = np.arange(0, 300, 10)
    plt.plot(xxx, FCMRes.Jlist, 'g-', )
    plt.xlabel('Iterations', fontproperties="SimHei")
    plt.ylabel('Objective function values', fontproperties="SimHei")
    plt.title("Convergence curve")

    # The fourth Pic (Routes of large UAV and small UAV)
    plt.subplot(234)
    endUAVRoutes = generateEndUAVRoute(FCMRes)
    for key, valule in endUAVRoutes:
        plt.plot(valule[0], valule[1], label='路线', linewidth=2, marker='o', markersize=4)

    MotherUAVPonit = np.zeros((len(endUAVRoutes),2))
    for i, endUAV in endUAVRoutes:
        MotherUAVPonit[i][0] = endUAV[0][-1]
        MotherUAVPonit[i][1] = endUAV[1][-1]

    city_location = FCMRes.Clast
    N, record, mile_cost, satisfactory_solution = generateMotherUAVRoute(MotherUAVPonit)
    # Draw routes
    mX = []
    mY = []
    for i in satisfactory_solution:
        x = MotherUAVPonit[i - 1][0]
        y = MotherUAVPonit[i - 1][1]
        mX.append(x)
        mY.append(y)
    plt.plot(mX, mY, '-o', color='b',linewidth=3)
    plt.plot(FCMRes.airports[:, 0], FCMRes.airports[:, 1], 'r*', markersize=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Route planning for HUAVs" )

    plt.show()

# Loading instances data
def loadData(filename, N):
    file = open(filename)
    lines = file.readlines()
    datas = np.zeros((N, 2))
    row_count = 0
    for line in lines:
        line = line.strip().split('\t')
        datas[row_count, :] = line[:]
        row_count += 1
    return datas

# Routing the large UAVs
def generateMotherUAVRoute(city_location):
    N = 100                       # Iteration times of tabu search algorithm
    time_start = time.time()
    # The cluster center point is selected as the track point of large UAV
    vns_ts = VNS_TS(city_location)
    satisfactory_solution, mile_cost, record = vns_ts.tabu_search(vns_ts.remain_cities, N)
    time_end = time.time()
    time_cost = time_end - time_start
    print('time cost:',time_cost)
    print("Costs:%d" %(int(mile_cost)))
    print("Optimization Routes:\n", satisfactory_solution)
    return N, record, mile_cost, satisfactory_solution


if __name__ == '__main__':

    datafilename = 'Data/Random/data50.txt'
    datas = loadData(datafilename, 55)
    cities = datas[0:50]
    airports = datas[50:55]
    FCMRes = FCM(cities, airports, 10, 30)
    plot(FCMRes)
 