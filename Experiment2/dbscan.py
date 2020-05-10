import numpy as np

import matplotlib.pyplot as plt
import queue

unassigned = -1
noise = 0
"""
加载数据集  id  latitude longitude
"""
def load_dataset(filename):
    r = open(filename, 'r')
    datas = r.readlines()
    list = []
    del (datas[0])
    for i in range(0, len(datas)):
        temp_list = datas[i].split(',')
        temp_list2 = []
        # temp_list2.append(float(temp_list[0]))
        temp_list2.append(float(temp_list[1]))
        temp_list2.append(float(temp_list[2]))
        list.append((float(temp_list[1]), float(temp_list[2])))
        if i > 1000:
            break
    return list
"""
计算两个向量的距离
"""
import math
def dist(vec1, vec2):
    return math.sqrt(np.power(vec1 - vec2, 2).sum())
    # return  np.sqrt(np.sum(np.square(vec1 - vec2)))


"""
得到邻域内所有样本点的Id
data: 样本点
core_point_id  核心点
radius  半径
"""
def neighbor_points(data, core_point_id, radius):
    points = []
    for i in range(len(data)):
        #计算 每个点与核心点的距离  如果小于半径  则加入此簇
        if dist(data[i,1:3], data[core_point_id,1:3]) < radius:
            points.append(i)
    return np.asarray(points)

"""
判断一个点是否是核心点，若是则将它和它邻域内的所用未分配的样本点分配给一个新类
若邻域内有其他核心点，重复上一个步骤，但只处理邻域内未分配的点，并且仍然是上一个步骤的类。
    data: 样本集合
     cluster_result: 聚类结果
     point_id:  样本Id
     cluster_id  类Id
     radius   半径
     minPts   最小局部密度
"""

def to_cluster(data, cluster_result, point_id, cluster_id, radius, minPts):
    # 计算以point_id为核心的点 有多少  以point_id为圆心   以radius为半径
    points = neighbor_points(data, point_id, radius)
    points = points.tolist()

    q = queue.Queue()

    #如果以point_id 为核心的范围内的点数 小于minPts  则说明 point_id不能作为核心
    if len(points) < minPts:
        cluster_result[point_id] = noise
        return False
    else:
        #否则将这个点作为 簇cluster_id的核心
        cluster_result[point_id] = cluster_id
#使用这一点形成一个新的聚类 ，并包括集群内的邻域内或边界上的所有点。
    for point in points:
        #如果这个点还没有被分配到任何一个类
        if cluster_result[point] == unassigned:
            q.put(point)
            #则将其分配
            cluster_result[point] = cluster_id
        #当队列不为空
    while not q.empty():
        #从队列中删除一个点 计算这个点的领域内的点
        neighbor_result = neighbor_points(data, q.get(), radius)
            #如果这个点邻域内的点数 大于 minPts 则说明是核心点
        if len(neighbor_result) >= minPts:                      # 核心点
            for i in range(len(neighbor_result)):
                result_point = neighbor_result[i]
                #如果这个点没有被分配
                if cluster_result[result_point] == unassigned:
                    q.put(result_point)
                    #则分配一个簇
                    cluster_result[result_point] = cluster_id
                    #如果这个点是噪声点  则
                elif cluster_result[cluster_id] == noise:
                    cluster_result[result_point] = cluster_id
    return True
"""
扫描整个数据集，为每个数据集打上核心点，边界点和噪声点标签的同时为 样本集聚类
data: 样本集
radius: 半径
minPts:  最小局部密度
"""
def dbscan(data, radius, minPts):
    cluster_id = 1 #簇标号
    points_size = len(data)  #点的数量
    cluster_result = [unassigned] * points_size
    for id in range(points_size):
        if cluster_result[id] == unassigned:
            if to_cluster(data, cluster_result, id, cluster_id, radius, minPts):
                cluster_id = cluster_id + 1
                print(cluster_id)
    return np.asarray(cluster_result), cluster_id

def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')

data_list = load_dataset('data/temp.csv')
cluster = np.asarray(data_list)
clusterRes, clusterNum = dbscan(cluster, 0.05, 20)
print(clusterRes)
print(clusterNum)
plotRes(cluster, clusterRes, clusterNum)
# nmi, acc, purity = eva.eva(clusterRes, cluster)
# print(nmi, acc, purity)
plt.show()