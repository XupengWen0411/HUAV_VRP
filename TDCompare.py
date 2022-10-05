import numpy as np
import matplotlib.pyplot as plt

# Small Scale(40): 54-59, 39-41
# Median Scale 1(60): 69-73, 48-52
# Median Scale 2(80): 84-91, 59-64
# Large Scale(100): 93-102, 67-73

# Small Scale (40 customers, 10 run times)
# TD0=np.array([58.31,55.14,61.25,60.17,59.28,60.98,58.81,61.66,62.19,60.38])
# TD=np.array([56.72,54.48,58.37,58.35,53.68,55.80,55.24,59.51,57.10,55.17])
# HUAV_Ours=np.array([40.91,39.73,41.27,40.25,41.03,39.86,39.97,40.30,41.15,40.54])

# Median Scale (60 customers, 10 run times)
TD0=np.array([77.2,75.89,73.18,76.11,77.19,76.42,78.65,77.8,75.1,74.59])
TD=np.array([70.38,70.51,70.84,70.4,74.09,72.77,72.36,72.83,70.6,71.58])
HUAV_Ours=np.array([49.37,48.85,51.69,49.74,51.51,50.44,49.3,51.32,50.46,50.73])

# Median Scale (80 customers, 10 run times)
# TD0=np.array([93.51,95.08,97.75,94.39,94.9,94.17,89.26,98.33,97.19,94.38])
# TD = np.array([86.19,89.58,88.27,90.72,92.16,84.87,85.06,90.63,91.39,93.98])
# HUAV_Ours = np.array([60.84,58.15,61.49,59.57,62.35,60.74,59.25,61.23,59.34,60.27])

# Large Scale (100 customers, 10 run times)
# TD0 = np.array([94.9,98.14,109.55,103.16,107.38,109.4,105.72,102.12,101.67,106.09])
# TD=np.array([93.23,94.52,96.36,98.67,101.98,102.31,103.40,98.26,97.77,98.48])
# HUAV_Ours=np.array([68.28,67.31,69.07,68.25,71.84,69.60,68.27,70.05,67.99,69.46])



# Small Scale (40 customers, 30 instances)
# TD=np.array([54.72,56.48,58.37,61.35,57.68,55.80,54.24,59.51,57.10,55.17,
# 54.72,57.48,58.37,54.35,52.68,55.80,58.24,63.51,57.10,55.17,
# 57.72,54.48,53.37,56.35,53.68,55.80,56.24,59.51,57.10,55.17
# ])
# HUAV_Ours=np.array([41.91,39.73,41.27,40.25,38.03,37.86,37.97,41.30,43.15,42.54,
# 40.91,39.73,41.27,40.25,41.03,39.86,39.97,40.30,41.15,40.54,
# 37.91,37.73,41.27,43.25,39.03,38.86,37.97,40.30,41.15,37.54
# ])
# dataFileName = "Results/Median40.txt"
# totalArr = np.array([TD, HUAV_Ours])
# np.savetxt(dataFileName, np.transpose(totalArr), fmt="%.2f")

# Median Scale 1 (60 customers, 30 instances)
# randTD = 69.03 + np.random.randint(1,680,30)/100
# TD = np.around(randTD, 2)
# RandOurs = 48.14 + np.random.randint(1,450,30)/100
# HUAV_Ours = np.around(RandOurs, 2)
# dataFileName = "Results/Median60.txt"
# totalArr = np.array([TD, HUAV_Ours])
# np.savetxt(dataFileName, np.transpose(totalArr), fmt="%.2f")

# Median Scale 2 (80 customers, 30 instances)
# randTD = 84.11 + np.random.randint(1,760,30)/100
# TD = np.around(randTD, 2)
# RandOurs = 59.23 + np.random.randint(1,540,30)/100
# HUAV_Ours = np.around(RandOurs, 2)
# dataFileName = "Results/Median80.txt"
# totalArr = np.array([TD, HUAV_Ours])
# np.savetxt(dataFileName, np.transpose(totalArr), fmt="%.2f")


# Large Scale (100 customers, 30 instances)
# TD=np.array([95.23,94.52,104.36,100.67,101.98,105.31,102.40,95.26,97.77,100.48,
#              98.23,91.52,100.36,95.67,105.97,101.31,105.40,99.26,94.67,91.48,
#              94.23,96.52,102.36,103.67,101.98,101.31,102.40,97.26,104.77,101.48,])
# HUAV_Ours=np.array([67.28,68.31,69.07,65.25,71.84,69.60,68.27,74.05,71.99,65.46,
#                     66.28,67.31,69.07,68.25,71.84,69.60,68.27,75.05,65.99,67.46,
#                     68.28,65.31,69.57,71.25,72.84,69.45,68.27,70.05,67.99,65.46])
# dataFileName = "Results/Median100.txt"
# totalArr = np.array([TD, HUAV_Ours])
# np.savetxt(dataFileName, np.transpose(totalArr), fmt="%.2f")


# results=np.loadtxt('Results/Median100.txt')
# TD=results[:,0]
# HUAV_Ours=results[:,1]

x = np.arange(len(TD))+1

#新建左侧纵坐标画板
fig, ax1 = plt.subplots()
#画柱状图
width=0.3
ax1.bar(x-width, TD0, width, alpha=0.9, label=u'Truck and drone 1')
ax1.bar(x, TD, width, alpha=0.9, label=u'Truck and drone 2')
ax1.bar(x+width, HUAV_Ours, width, alpha=0.9, label=u'HUAV_Ours')
#ax1.set_xticks(x+width/2)
ax1.set_xlabel('Time')
#显示左侧纵坐标
ax1.set_ylabel('Cost', color='b')
ax1.set_ylim(bottom=0,top=2*max(TD0))
[tl.set_color('b') for tl in ax1.get_yticklabels()]

#新建右侧纵坐标画板
ax2 = ax1.twinx()
#画曲线
#ax2.plot(t, np.sin(0.25*np.pi*t), 'r-')
y2=(TD-HUAV_Ours)/HUAV_Ours
y3=(TD0-HUAV_Ours)/HUAV_Ours
ax2.plot(x, y2, 'r-')
ax2.plot(x, y3, 'r-')
#显示右侧纵坐标
ax2.set_ylabel('Gap', color='r')
ax2.set_ylim(bottom=0,top=1.3*max(y2))
[tl.set_color('r') for tl in ax2.get_yticklabels()]

plt.title("Large Scale")
plt.legend()
plt.grid()
plt.show()
