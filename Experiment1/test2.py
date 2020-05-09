
def getData(filename):
    f = open(filename,'r')
    datas = f.readlines()
    label = datas[0].split(',')
    del (label[6])
    # label[len(label)-1] = label[len(label)-1].replace('\n','')
    data_set = []
    for i in range(1,len(datas)):
        temp = datas[i].split(',')
        temp[len(temp) - 1] = temp[len(temp) - 1].replace('\n', '')
        data_set.append(temp)
    return label,data_set

label_set,data_set = getData('data/car_data.csv')

print(label_set)
#加载训练数据 80% 测试数据 20%
train_list = data_set[0:1382]  # 训练集 数据
test_list = data_set[1383:]  # 测试集 数据
print('训练集大小',len(train_list))
print('测试集大小',len(test_list))

label_count = {'buying':4,'maint':4,'doors':4,'persons':3,'lug_boot':3,'safety':3}
import operator
def training(features, targets):
    # 计算各类别的样本个数
    label_count = {}
    for classify in targets:
        if classify not in label_count.keys():
            label_count[classify] = 0
        label_count[classify] += 1
    print(label_count)

    k = len(label_count.keys())  # 类别数
    # feature 的 m行  n列
    m = len(features)
    n = len(features[0])

    # 计算先验概率  每一类出现的概率
    lamb = 1  # 平滑处理的参数  拉普拉斯修正
    prior = dict()  # 存储先验概率

    for label, amount in label_count.items():
        prior[label] = (amount + lamb) / (m + k * lamb)  # 计算平滑处理后的先验概率 拉普拉斯修正

    print(prior)

    conditional = dict()  # 存储条件概率
    for feature in range(n):  # 遍历每个特征
        conditional[feature] = {}
        values = []  # 存放 每一个特征有多少种取值
        for data in features:
            if data[feature] not in values:
                values.append(data[feature])
        for value in values:  # 遍历每个特征值
            conditional[feature][value] = {}  # 第i的特征的取值value
            for label, amount in label_count.items():  # 遍历每种类别
                # 截取该类别的数据集
                feature_label = []  # 存储类别为label的数据集
                for j in range(0, m):
                    if label == targets[j]:
                        feature_label.append(features[j])
                # 计算该类别下各特征值出现的次数
                c_label = {}  # 存放该类别下 每一个特征值出现的次数
                for data in feature_label:
                    if data[feature] not in c_label.keys():
                        c_label[data[feature]] = 0
                    c_label[data[feature]] += 1
                # 计算平滑处理后的条件概率  拉普拉斯修正
                conditional[feature][value][label] = (c_label.get(value, 0) + lamb) / \
                                                     (amount + len(values) * lamb)  # 计算平滑处理后的条件概率

    return prior, conditional


features = []  # 数据在每一纬度的特征   特征集m*n,m为样本数,n为特征数
target = []  # 数据的真实分类   标签集
for data in train_list:
    features.append(data[0:-1])
    target.append(data[-1])
prior, conditional = training(features, target)  # 计算 先验概率  条件概率

import numpy as np


def predict(test, prior, conditional):
    """预测单个样本"""
    best_poster, best_label = -np.inf, -1
    for label in prior:
        # 初始化后验概率为先验概率,同时把连乘换成取对数相加，防止下溢（即太多小于1的数相乘，结果会变成0）
        poster = np.log(prior[label])
        for i in range(0, len(test)-1):
            # 第i个特征（第i列） 取值为test[i]  类别为label 的条件概率   由于取了对数 所以加变成了乘
            poster += np.log(conditional[i][test[i]][label])
        if poster > best_poster:  # 获取后验概率最大的类别
            best_poster = poster
            best_label = label
    return best_label

#
# test = ['3','1','2','4','1','1','1']
#
# classify_result = predict(test, prior, conditional)
# print(classify_result)
pos_count = 0  # 正确分类数量
neg_count = 0  # 错误分类数量
classify_result_list = []  # 存储分类结果
for test in test_list:
    classify_result = predict(test, prior, conditional)
    classify_result_list.append(classify_result)

for i in range(0, len(classify_result_list)):
    if classify_result_list[i] == test_list[i][6]:
        pos_count += 1
    else:
        neg_count += 1
print('测试集总数', len(test_list))
print('正确分类数', pos_count)
print('错误分类数', neg_count)
print('分类的准确率为', pos_count / len(test_list))