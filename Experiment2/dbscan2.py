import numpy as np
import matplotlib.pyplot as plt
import queue
from geopy.distance import geodesic,distance
"""
加载数据集    latitude longitude
"""
def load_dataset(filename,length):
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
        if i > length:
            break
    return list
"""
计算两个向量的距离
"""
def dist(vec1, vec2):
    # dist = distance(vec1, vec2).miles
    dist = np.sqrt(np.sum(np.square(np.array(vec1) - np.array(vec2))))
    return dist
"""
得到邻域内所有样本点的Id
data: 样本点
core_point_id  核心点
radius  半径
"""
def neighbor_points(data, core_point, radius):
    points = []
    for i in range(0,len(data)):
        #计算 每个点与核心点的距离  如果小于半径  则加入此簇
        if type(core_point).__name__ == 'tuple':
            if dist(data[i], core_point) < radius:
                points.append(data[i])
        else:
            if dist(data[i], data[core_point]) < radius:
                points.append(data[i])
    return points

"""
扫描整个数据集，为每个数据集打上核心点，
data: 样本集
radius: 半径
minPts:  最小局部密度
"""
import random
def dbscan(datas, radius, minPts):
    # 聚类个数
    k = 0
    # 核心对象集合
    omega = set()
    # 未访问样本集合
    not_visit = set(datas)
    # 聚类结果
    cluster = dict()
    cluster_core = []
    # 遍历样本集找出所有核心对象
    for point_id in range(0,len(datas)):
        # 计算以point_id为核心的点 有多少  以point_id为圆心   以radius为半径
        points = neighbor_points(datas, point_id, radius)
        if len(points) >= minPts:
            omega.add(datas[point_id])
    #遍历核心对象的集合
    while len(omega):
        # 记录当前未访问样本集合
        not_visit_old = not_visit
        # 随机选取一个核心对象core
        core = list(omega)[random.randint(0, len(omega)-1)]
        print(core)
        cluster_core.append(core)
        not_visit  = not_visit - set(core)
        # 初始化队列，存放核心对象或样本
        core_deque = queue.Queue()
        core_deque.put(core)
        while not core_deque.empty():
            coreq = core_deque.get()
            # 找出以coreq邻域内的样本点
            coreq_neighborhood = neighbor_points(datas, coreq, radius)
            # 若coreq为核心对象，求其邻域内且未被访问过的样本找出
            if len(coreq_neighborhood) >= minPts:
                intersection = set()
                for i in coreq_neighborhood:
                    if i in  list(not_visit) :
                        intersection.add(i)
                #将领域内未被访问过的样本找到 加入队列  对聚类进行扩散
                for i in list(intersection):
                    core_deque.put(i)
                    #将这些点标记为已访问
                not_visit  = not_visit - intersection
        #当队列再次为空时，一个聚类已经形成

        #这个聚类内的点
        Ck = not_visit_old - not_visit
        omega = omega - Ck
        cluster[k] = list(Ck)
        k +=1
    return cluster,cluster_core

def show_dataset(dataset):
    for item in dataset:
        plt.scatter(item[0], item[1], c='red', alpha=1, marker='+')
    plt.title("Dataset")
    print('Dataset done')
    plt.show()

def plotRes( clusterRes):

    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown','m', 'fuchsia', 'crimson', 'dodgerblue', 'lime', 'coral', 'peru', 'khaki','black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown','m', 'fuchsia', 'crimson', 'dodgerblue', 'lime', 'coral', 'peru', 'khaki','black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown','m', 'fuchsia', 'crimson', 'dodgerblue', 'lime', 'coral', 'peru', 'khaki']
    for key in clusterRes.keys():
        for item in clusterRes[key]:

                plt.scatter(item[0], item[1], c=scatterColors[key], alpha=1, marker='+')
    plt.title("DBSCAN Clustering")
    plt.show()
data_list = load_dataset('data/temp.csv',500)
# show_dataset(data_list)
print(500," ","0.7"," ",25)
clusterRes,cluster_core = dbscan(data_list, 0.35, 10)
print(len(clusterRes.keys()))
print(len(cluster_core))
plotRes(clusterRes)


# 聚类平均距离
def avg_dis(cluster):
    sum = 0.0
    for i in cluster:
        for j in cluster:
            sum += dist(i, j)
    l = len(cluster)  # 聚类点的个数
    return sum / (l * (l - 1))


def compute_Rij(i, j, k, cluster_res, cluster_core):
    avg_ci = avg_dis(cluster_res[i])
    avg_cj = avg_dis(cluster_res[j])
    dcen = dist(cluster_core[i], cluster_core[j])  # 两个聚类核心点的距离
    res = (avg_ci + avg_cj) / dcen
    return res


def compute_max(index, cluster_res, k, cluster_core):
    list_r = []
    for j in range(0, k):
        if index != j:
            temp = compute_Rij(index, j, k, cluster_res, cluster_core)
            list_r.append(temp)
    return max(list_r)


def computer_db_index(cluster_res, cluster_core):
    k = len(cluster_core)  # 聚类总数
    sigma_R = 0.0
    for i in range(0, k):
        sigma_R += compute_max(i, cluster_res, k, cluster_core)
    dbi = float(sigma_R) / float(k)
    return dbi


dbi = computer_db_index(clusterRes, cluster_core)