import math,random,time
import matplotlib.pyplot as plt

class VNS_TS:
    def __init__(self, city_location):
        self.city_location = city_location
        self.city_count = len(self.city_location)        # 总的城市数
        self.city_num = list(range(0, self.city_count))  # 城市编号
        self.dis = self.distance()
        self.origin = 1                                  # 设置起点和终点城市
        self.remain_cities = self.city_num[:]
        self.remain_cities.remove(self.origin)           # 迭代过程中变动的城市
        self.remain_count = self.city_count - 1
        self.improve_count = 100                         # 改良次数
        self.candiate_routes_size = 1.5*self.city_count  # 邻域规模(这个值需要根据客户点的数量来改变)
        self.tabu_size = 10
        self.tabu_list = []                              # 禁忌表
        self.best_so_far_cost = 0
        self.best_so_far_route = []
        self.tabu_list = []

    #计算距离邻接矩阵
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
        计算路径的里程
        '''
        mile_cost = 0.0
        mile_cost += self.dis[self.origin-1][route[0]-1]    # 从起始点开始
        for i in range(self.remain_count-1):                # 路径的长度
            mile_cost += self.dis[route[i]-1][route[i+1]-1]
        mile_cost += self.dis[route[-1]-1][self.origin-1]   # 到终点结束
        return mile_cost

    def random_initial_route(self,remain_cities):
        '''
        随机生成初始路径
        '''
        initial_route = remain_cities[:]
        random.shuffle(initial_route)
        mile_cost = self.route_mile_cost(initial_route)
        return initial_route,mile_cost

    def improve_circle(self,remain_cities):
        '''
        改良圈算法生成初始路径
        '''
        initial_route = remain_cities[:]
        random.shuffle(initial_route)
        cost0 = self.route_mile_cost(initial_route)
        route = [1] + initial_route + [1]
        label = list(i for i in range(1,len(remain_cities)))
        j = 0
        while j < self.improve_count:
            new_route = route[:]
            #随机交换两个点
            index0,index1 = random.sample(label,2)
            new_route[index0],new_route[index1]= new_route[index1],new_route[index0]
            cost1 = self.route_mile_cost(new_route[1:-1])
            improve = cost1 - cost0
            if improve < 0: #交换两点后有改进
                route = new_route[:]
                cost0 = cost1
                j += 1
            else:
                continue
        initial_route = route[1:-1]
        return initial_route,cost0

    #获取当前邻居城市中距离最短的1个
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
        采用贪婪算法生成初始解。从第一个城市出发找寻与其距离最短的城市并标记，
        然后继续找寻与第二个城市距离最短的城市并标记，直到所有城市被标记完。
        最后回到第一个城市(起点城市)
        '''
        cand_cities = remain_cities[:]
        current_city = self.origin
        mile_cost = 0
        initial_route = []
        while len(cand_cities) > 0:
            next_city,distance = self.nearest_city(current_city,cand_cities) #找寻最近的城市及其距离
            mile_cost += distance
            initial_route.append(next_city)             #将下一个城市添加到路径列表中
            current_city = next_city                    #更新当前城市
            cand_cities.remove(next_city)               #更新未定序的城市
        mile_cost += self.dis[initial_route[-1]-1][0]   #回到起点
        return initial_route,mile_cost

    def random_swap_2_city(self,route):
        '''
        随机选取两个城市并将这两个城市之间的点倒置,计算其里程成本
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
        生成新的路线
        '''
        candidate_routes = []                   #路线候选集合
        candidate_mile_cost = []                #候选集合路线对应的里程成本
        candidate_swap = []                     #交换元素
        while len(candidate_routes) < self.candiate_routes_size:
            cand_route,cand_swap = self.random_swap_2_city(route)
            if cand_swap not in candidate_swap: #此次生成新路线的过程中，没有被交换过
                candidate_routes.append(cand_route)
                candidate_swap.append(cand_swap)
                candidate_mile_cost.append(self.route_mile_cost(cand_route))
        min_mile_cost = min(candidate_mile_cost)
        i = candidate_mile_cost.index(min_mile_cost)
        #如果此次交换集的最优值比历史最优值更好，则更新历史最优值和最优路线
        if min_mile_cost < self.best_so_far_cost:
            self.best_so_far_cost = min_mile_cost
            self.best_so_far_route = candidate_routes[i]
            new_route = candidate_routes[i]
            if candidate_swap[i] in self.tabu_list:
                self.tabu_list.remove(candidate_swap[i]) #藐视法则
            elif len(self.tabu_list) >= self.tabu_size:
                self.tabu_list.remove(self.tabu_list[0])
            self.tabu_list.append(candidate_swap[i])
        else:
            #此次交换集未找到更优路径，则选择交换方式未在禁忌表中的次优
            K = self.candiate_routes_size
            stop_value = K - len(self.tabu_list) - 1
            while K > stop_value:
                min_mile_cost = min(candidate_mile_cost)
                i = candidate_mile_cost.index(min_mile_cost)
                #如果此次交换集的最优值比历史最优值更好，则更新历史最优值和最优路线
                if min_mile_cost < self.best_so_far_cost:
                    self.best_so_far_cost = min_mile_cost
                    self.best_so_far_route = candidate_routes[i]
                    new_route = candidate_routes[i]
                    if candidate_swap[i] in self.tabu_list:
                        self.tabu_list.remove(candidate_swap[i]) #藐视法则
                    elif len(self.tabu_list) >= self.tabu_size:
                        self.tabu_list.remove(self.tabu_list[0])
                    self.tabu_list.append(candidate_swap[i])
                    break
                else:
                    #此次交换集未找到更优路径，则选择交换方式未在禁忌表中的次优
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
        #生成初始解
        self.best_so_far_route,self.best_so_far_cost =self.greedy_initial_route(remain_cities)
        # best_so_far_route,best_so_far_cost = random_initial_route(remain_cities)
        # best_so_far_route,best_so_far_cost = improve_circle(remain_cities)
        record = [self.best_so_far_cost] #记录每一次搜索后的最优值
        new_route = self.best_so_far_route[:]
        #生成邻域
        for j in range(iteration_count):
            new_route = self.generate_new_route(new_route)
            record.append(self.best_so_far_cost)
        final_route = [self.origin] + self.best_so_far_route +[self.origin]
        return final_route,self.best_so_far_cost,record

    def plot(self, N, mile_cost, record, satisfactory_solution, city_location):
        # 绘制路线图
        X = []
        Y = []
        for i in satisfactory_solution:
            x = city_location[i - 1][0]
            y = city_location[i - 1][1]
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, '-o')
        plt.title("大无人机路径规划图:%d" % (int(mile_cost)))
        plt.show()
        # 绘制迭代过程图
        A = [i for i in range(N + 1)]   # 横坐标
        B = record[:]                   # 纵坐标
        plt.xlim(0, N)
        plt.xlabel('迭代次数', fontproperties="SimSun")
        plt.ylabel('路径里程', fontproperties="SimSun")
        plt.title("solution of TS changed with iteration")
        plt.plot(A, B, '-')
        plt.show()

#读取坐标文件
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
    N = 100         #迭代次数
    time_start = time.time()
    city_location = loadData()
    vns_ts = VNS_TS(city_location)
    satisfactory_solution, mile_cost, record = vns_ts.tabu_search(vns_ts.remain_cities, N)
    time_end = time.time()
    time_cost = time_end - time_start
    print('time cost:', time_cost)
    print("优化里程成本:%d" %(int(mile_cost)))
    print("优化路径:\n", satisfactory_solution)
    vns_ts.plot(N, mile_cost, record, satisfactory_solution, vns_ts.city_location)
