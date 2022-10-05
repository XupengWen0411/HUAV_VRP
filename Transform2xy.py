import math
import numpy as np


class Transform2xy:
    def __init__(self, datas):
        self.xy_coordinate = np.zeros((len(datas), 2))  # 转换后的XY坐标集
        self.datas = datas

    def transform(self):
        for i, data in enumerate(self.datas):
            ab=np.array(self.millerToXY(data[0], data[1]))
            self.xy_coordinate[i] = np.array(self.millerToXY(data[0], data[1]))
        return self.xy_coordinate

    def millerToXY(self, lon, lat):
        """
        经纬度转换为平面坐标系中的x,y 利用米勒坐标系
        :param lon: 经度
        :param lat: 维度
        :return:
        """
        L = 6381372*math.pi*2
        W = L
        H = L/2
        mill = 2.3
        x = lon*math.pi/180
        y = lat*math.pi/180
        y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))
        x = (W/2)+(W/(2*math.pi))*x
        y = (H/2)-(H/(2*mill))*y
        # self.xy_coordinate.append((int(round(x)),int(round(y))))

        return int(round(x)), int(round(y))
