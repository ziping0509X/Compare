import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DQN_l_path ='D:\YuanZihong\SensorModel\Loss.csv'
DQN_r_path ='D:\YuanZihong\SensorModel\Reward.csv'
DRQN_l_path = 'D:\YuanZihong\SensorModel_DRQN_1\Loss.csv'
DRQN_r_path = 'D:\YuanZihong\SensorModel_DRQN_1\Reward.csv'

DQN_l = pd.read_csv(DQN_l_path)
DQN_r = pd.read_csv(DQN_r_path)

DRQN_l = pd.read_csv(DRQN_l_path)
DRQN_r = pd.read_csv(DRQN_r_path)

DQN_l = np.array(DQN_l)
DQN_l_1 = []

for i in range(180000):
    DQN_l_1.append(DQN_l[i][1])


DRQN_l = np.array(DRQN_l)
DRQN_l_1 = []
len3 = len(DRQN_l)

for i in range(len3):
    DRQN_l_1.append(DRQN_l[i][1])

plt.rcParams["font.family"]="SimHei"
plt.rcParams['axes.unicode_minus']=False

fig = plt.figure(num= 1,figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.set_xlabel("迭代次数")
ax1.set_ylabel("损失函数")
ax1.set_title("DQN/DRQN损失函数")
ax1.plot(DQN_l_1,c='b',label="DQN")
ax1.plot(DRQN_l_1,c='c',label="DRQN")
ax1.legend(loc=1)

DQN_r = np.array(DQN_r)
DQN_r_1 = []
len2 = len(DQN_r)

for i in range(len2):
    DQN_r_1.append(DQN_r[i][1])

DRQN_r = np.array(DRQN_r)
DRQN_r_1 = []
len4 = len(DRQN_l)

for i in range(len4):
    DRQN_r_1.append(DRQN_r[i][1])

fig = plt.figure(num= 1,figsize=(12,6))
ax1 = fig.add_subplot(122)
ax1.set_xlabel("迭代次数")
ax1.set_ylabel("奖励函数")
ax1.set_title("DQN/DRQN奖励函数")
ax1.plot(DQN_r_1,c='b',label="DQN")
ax1.plot(DRQN_r_1,c='c',label="DRQN")
ax1.legend(loc=1)

plt.show()


