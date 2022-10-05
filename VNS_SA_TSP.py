import math,random,time
import matplotlib.pyplot as plt

#读取坐标文件
filename = 'berlin52.txt'
city_num = [] #城市编号
city_location = [] #城市坐标
with open(filename, 'r') as f:
    datas = f.readlines()[0:-1]
for data in datas:
    data = data.split()
    city_num.append(int(data[0]))
    x = float(data[1])
    y = float(data[2])
    city_location.append((x,y))#城市坐标

city_count = len(city_num) #总的城市数
origin = 1 #设置起点城市和终点城市
city_num.remove(origin)
remain_cities = city_num[:]  #迭代过程中变动的城市
remain_count = city_count - 1 #迭代过程中变动的城市数
#计算邻接矩阵
dis =[[0]*city_count for i in range(city_count)] #初始化
for i in range(city_count):
    for j in range(city_count):
        if i != j:
            dis[i][j] = math.sqrt((city_location[i][0]-city_location[j][0])**2 + (city_location[i][1]-city_location[j][1])**2)
        else:
            dis[i][j] = 0


def route_mile_cost(route):
    '''
    计算路径的里程成本
    '''
    mile_cost = 0.0
    mile_cost += dis[origin-1][route[0]-1]#从起始点开始
    for i in range(remain_count-1):#路径的长度
        mile_cost += dis[route[i]-1][route[i+1]-1]
    mile_cost += dis[route[-1]-1][origin-1] #到终点结束
    return mile_cost

#获取当前邻居城市中距离最短的1个
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
    采用贪婪算法生成初始解：从第一个城市出发找寻与其距离最短的城市并标记，
    然后继续找寻与第二个城市距离最短的城市并标记，直到所有城市被标记完。
    最后回到第一个城市(起点城市)
    '''
    cand_cities = remain_cities[:]
    current_city = origin
    initial_route = []
    mile_cost = 0
    while len(cand_cities) > 0:
        next_city,distance = nearest_city(current_city,cand_cities) #找寻最近的城市及其距离
        mile_cost += distance
        initial_route.append(next_city) #将下一个城市添加到路径列表中
        current_city = next_city #更新当前城市
        cand_cities.remove(next_city) #更新未定序的城市
    mile_cost += dis[initial_route[-1]-1][origin-1] #回到起点
    return initial_route,mile_cost

indexs = list(k for k in range(remain_count))
def random_swap_2_city(route):
    '''
    随机选取两个城市并将这两个城市之间的数据点倒置,生成新的回路
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
    record = [best_value] #记录温度下降对应的当前解
    while T > Tfloor:
        for i in range(iter_count):
            cand_route = random_swap_2_city(route)
            cand_mile = route_mile_cost(cand_route)
            if cand_mile <= mile: #如果候选解比当前最优解更优则更新当前最优解
                route = cand_route[:]
                mile = cand_mile
                best_value = mile
                best_route = route
                # T -= mile - cand_mile
            else: #如果候选解比当前最优解差，则按概率接受候选解
                p = math.exp((mile - cand_mile)/T)
                if random.random() < p:
                    route = cand_route[:]
                    mile = cand_mile
        T *= alpha #降温
        record.append(best_value)
    best_route = [origin] + best_route + [origin]
    return best_route,best_value,record
def fig():
    time_start = time.time()
    satisfactory_solution,mile_cost,record = main()
    time_end = time.time()
    time_cost = time_end - time_start
    print('time cost:',time_cost)
    print("优化里程成本:%d" %(int(mile_cost)))
    print("优化路径:\n",satisfactory_solution)
    #绘制路线图
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
    #绘制迭代过程图
    plt.xlabel('温度变化',fontproperties="SimSun")
    plt.ylabel('路径里程',fontproperties="SimSun")
    plt.title("solution of SA changed with temperature")
    plt.plot(record,'-')
    plt.show()
    return mile_cost,time_cost
fig()
# R = 10
# Mcosts = [0]*R
# Tcosts = [0]*R
# for j in range(R):
#     Mcosts[j],Tcosts[j] = fig()
# AM = sum(Mcosts)/R #平均里程
# AT = sum(Tcosts)/R #平均时间
# print("最小里程:",min(Mcosts))
# print("平均里程:",AM)
# print('里程:\n',Mcosts)
# print("平均时间:",AT)
