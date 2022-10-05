import numpy as np
import matplotlib.pyplot as plt
import time
from Transform2xy import Transform2xy
from FCMCluster import FCM
from VNS_TS_TSP import VNS_TS
from DynamicTSP import DynaticTSP
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def dynamicTSPSolve(cities, distence_matrix):
    S = DynaticTSP(cities, distence_matrix, 0)
    print("最短距离：" + str(S.tsp()))
    # 开始回溯
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


def generateEndUAVRoute(FCMRes):
    Clast = FCMRes.Clast
    dict_cluster = FCMRes.dict_cluster
    airports = FCMRes.airports

    airportselect = []
    for cc in Clast:    # 聚类中心找到airport最近的点
        tempdis = 10000000
        index = -1
        for i in range(len(airports)):
            temp = (cc[0]-airports[i][0])**2 + (cc[1]-airports[i][1])**2
            if temp < tempdis:
                tempdis = temp
                index = i
        if index != -1:
            airportselect.append(index)

    # 规划小无人机的子路径
    endUAVRoutes = []
    for key in dict_cluster:
        cities = FCMRes.data[dict_cluster[key]]
        # cities = np.append(cities, np.array([airports[airportselect[key]]])) # 将所选择的airport点加到小无人机子路径中
        cities = np.insert(cities, 0, np.array([airports[airportselect[key]]]),0) # 将所选择的airport点加到小无人机子路径中第一个点
        # cities = np.append(cities, np.array([airports[airportselect[key]]]))  # 将聚类中心的点加到小无人机路径中
        cities = cities.reshape(len(dict_cluster[key]) + 1, 2)
        distence_matrix = FCMRes.distance(cities)
        xRoute, yRoute = dynamicTSPSolve(cities, distence_matrix)
        endUAVRoutes.append([key, [xRoute, yRoute]])
    return endUAVRoutes


def plot(FCMRes):
    mark = ['or', 'ob', 'og', 'om', 'oy', 'oc', 'sr', 'sb', 'sg', 'sm', 'sy', 'sc']
    # 第一张图（未聚类前散点图）
    plt.subplot(231)
    data = FCMRes.data
    plt.plot(data[:, 0], data[:, 1], 'ob', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('未聚类前散点图')
    plt.title('Scatter plot before clustering')

    # 第二张图（聚类后结果图）
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
    # plt.title("聚类后结果")
    plt.title("Clustering results")


    # 第三张图(聚类函数值变化图)
    plt.subplot(233)
    xxx = np.arange(0, 800, 10)
    plt.plot(xxx, FCMRes.Jlist, 'g-', )
    # plt.title("聚类方案调整图")
    # plt.xlabel('迭代次数', fontproperties="SimHei")
    # plt.ylabel('聚类目标函数值', fontproperties="SimHei")
    plt.xlabel('Iterations', fontproperties="SimHei")
    plt.ylabel('Objective function values', fontproperties="SimHei")
    plt.title("Convergence curve")


    # 第四张图(小无人机路径规划图)
    plt.subplot(234)
    endUAVRoutes = generateEndUAVRoute(FCMRes)
    for key, valule in endUAVRoutes:
        plt.plot(valule[0], valule[1], label='路线', linewidth=2, marker='o', markersize=4)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # # plt.title('小无人机路径规划图')
    # plt.title('Route planning for eUAV')
    # #plt.legend()
    #
    # # 第五张图(大无人机路径规划图)
    # plt.subplot(235)
    MotherUAVPonit = np.zeros((len(endUAVRoutes),2))
    for i, endUAV in endUAVRoutes:
        MotherUAVPonit[i][0] = endUAV[0][-1]
        MotherUAVPonit[i][1] = endUAV[1][-1]

    city_location = FCMRes.Clast
    N, record, mile_cost, satisfactory_solution = generateMotherUAVRoute(MotherUAVPonit)
    # 绘制路线图
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
    # plt.title("异构多无人机路径规划结果" )
    plt.title("Route planning for HUAVs" )


    # 第六张图(绘制大无人机路径规划迭代过程图)
    # plt.subplot(236)
    # A = [i for i in range(N + 1)]   # 横坐标
    # B = record[:]                   # 纵坐标
    # plt.xlim(0, N)
    # plt.xlabel('迭代次数', fontproperties="SimHei")
    # plt.ylabel('路径长度', fontproperties="SimHei")
    # plt.title("大无人机路径规划收敛图")

    # plt.xlabel('Iterations', fontproperties="SimHei")
    # plt.ylabel('Path length', fontproperties="SimHei")
    # plt.title("Convergence curve of routing")
    # plt.plot(A, B, '-')
    plt.show()


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


def generateMotherUAVRoute(city_location):
    N = 100                       # 禁忌搜索算法迭代次数
    time_start = time.time()
    # 大无人机的航迹点是选择的聚类点
    vns_ts = VNS_TS(city_location)
    satisfactory_solution, mile_cost, record = vns_ts.tabu_search(vns_ts.remain_cities, N)
    time_end = time.time()
    time_cost = time_end - time_start
    print('time cost:',time_cost)
    print("优化里程成本:%d" %(int(mile_cost)))
    print("优化路径:\n", satisfactory_solution)
    return N, record, mile_cost, satisfactory_solution


if __name__ == '__main__':

    datafilename = 'Data/Random/data50.txt'  # 自动机场的坐标也加到data里面去了
    datas = loadData(datafilename, 55)
    cities = datas[0:50]
    airports = datas[50:55]
    FCMRes = FCM(cities, airports, 10, 30)
    plot(FCMRes)
 