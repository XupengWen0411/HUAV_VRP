import numpy as np
import math


class FCM:
    def __init__(self, data, airports, clust_num, iter_num=10):
        self.data = data
        self.airports = airports
        self.cnum = clust_num
        self.sample_num=data.shape[0]
        self.dim = data.shape[-1]
        self.m = 2
        self.dict_cluster = []
        Jlist=[]
        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num):            #
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            print("The %d th iterations" %(i+1) ,end="")
            print("Cluster center",C)
            J = self.J_calcu(self.data, U, C)
            Jlist = np.append(Jlist, J)
        self.label = np.argmax(U, axis=0)       # Clustering label
        self.Clast = C                          # Cluster center matrix
        self.Jlist = Jlist                      # Objective function value matrix

    # Initialize membership matrix U
    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)
        row_sum = np.sum(U, axis=1)
        row_sum = 1 / row_sum
        U = np.multiply(U.T, row_sum)
        return U

    # Calculate cluster center
    def Cen_Iter(self, data, U, cluster_n):
        c_new = np.empty(shape=[0, self.dim])
        for i in range(0, cluster_n):
            u_ij_m = U[i, :] ** self.m
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)
            ux = np.reshape(ux, (1, self.dim))
            c_new = np.append(c_new, ux / sum_u, axis=0)
        return c_new                            # cluster_num*dim

    # Iteration of membership matrix
    def U_Iter(self, U, c):
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /
                            np.linalg.norm(self.data[j, :] - c[k, :])) ** (2 / (self.m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum
        return U

    # Calculate cluster cost value
    def J_calcu(self, data, U, c):
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** self.m
        J = np.sum(np.sum(temp1))
        return J

    # Distance matrix calculation
    def distance(self, city_location):
        city_count = len(city_location)
        dis = [[0] * city_count for i in range(city_count)]
        for i in range(city_count):
            for j in range(city_count):
                if i != j:
                    dis[i][j] = math.sqrt((city_location[i][0] - city_location[j][0]) ** 2 + (
                                city_location[i][1] - city_location[j][1]) ** 2)
                else:
                    dis[i][j] = 0
        return dis

    #  cluster Circle
    def clusterCircle(self):
        typeIndex = []
        for type in self.label:
            index = [x for x in range(len(self.label)) if self.label[x] == type]
            typeIndex.append([type, index])
        self.dict_cluster= dict(typeIndex)
        maxDisA = []
        for key in self.dict_cluster:
            subCluster = []
            sumX = 0
            sumY = 0
            for d in self.dict_cluster[key]:
                subCluster.append(self.data[d])
                sumX = sumX + self.data[d][0]
                sumY = sumY + self.data[d][1]
            averageX = sumX / len(self.dict_cluster[key])
            averageY = sumY / len(self.dict_cluster[key])
            dis = self.distance(subCluster)
            dis.sort()
            dis[-1].sort()
            maxDis = dis[-1][-1]
            maxDisA.append([key, [maxDis, averageX, averageY]])
        return maxDisA

    # Adjust Cluster By Weight
    def adjustClusterByWeight(self, cluster, weight, maximumLoad):
        sumWeight = sum(weight)
        if (sumWeight < maximumLoad):
            dis=[]
            for c in cluster:
                for od in self.data:
                    dis = math.sqrt((c[0] - od[0]) ** 2 + (c[1] - od[1]) ** 2)
