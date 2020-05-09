import numpy as np
from collections import Counter
from sklearn import datasets

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


class NaiveBayes:
    def __init__(self, lamb=1):
        self.lamb = lamb  # 贝叶斯估计的参数
        self.prior = dict()  # 存储先验概率
        self.conditional = dict()  # 存储条件概率

    def training(self, features, target):
        """
        根据朴素贝叶斯算法原理,使用 贝叶斯估计 计算先验概率和条件概率
        特征集集为离散型数据,预测类别为多元.  数据集格式为np.array
        :param features: 特征集m*n,m为样本数,n为特征数
        :param target: 标签集m*1
        :return: 不返回任何值,更新成员变量
        """
        features = np.array(features)
        target = np.array(target).reshape(features.shape[0], 1)
        m, n = features.shape

        labels = Counter(target.flatten().tolist())  # 计算各类别的样本个数

        k = len(labels.keys())  # 类别数
        for label, amount in labels.items():
            self.prior[label] = (amount + self.lamb) / (m + k * self.lamb)  # 计算平滑处理后的先验概率
        for feature in range(n):  # 遍历每个特征
            self.conditional[feature] = {}

            values = np.unique(features[:, feature])

            for value in values:  # 遍历每个特征值
                self.conditional[feature][value] = {}
                print(self.conditional)
                for label, amount in labels.items():  # 遍历每种类别
                    feature_label = features[target[:, 0] == label, :]  # 截取该类别的数据集
                    c_label = Counter(feature_label[:, feature].flatten().tolist())  # 计算该类别下各特征值出现的次数
                    print(c_label)
                    self.conditional[feature][value][label] = (c_label.get(value, 0) + self.lamb) / \
                                                              (amount + len(values) * self.lamb)  # 计算平滑处理后的条件概率
        return

    def predict(self, features):
        """预测单个样本"""
        best_poster, best_label = -np.inf, -1
        for label in self.prior:
            poster = np.log(self.prior[label])  # 初始化后验概率为先验概率,同时把连乘换成取对数相加，防止下溢（即太多小于1的数相乘，结果会变成0）
            for feature in range(features.shape[0]-1):
                poster += np.log(self.conditional[feature][features[feature]][label])
            if poster > best_poster:  # 获取后验概率最大的类别
                best_poster = poster
                best_label = label
        return best_label

import numpy as np
def test():
    # dataset = datasets.load_iris()  # 鸢尾花数据集
    # dataset = np.concatenate((dataset['data'], dataset['target'].reshape(-1, 1)), axis=1)  # 组合数据
    # np.random.shuffle(dataset)  # 打乱数据
    datasets = np.array(train_list)
    features = datasets[:, :-1]

    target = datasets[:, -1:]
    nb = NaiveBayes()
    nb.training(features, target)
    prediction = []

    for test in test_list:
        prediction.append(nb.predict(np.array(test)))
    pos_count = 0
    neg_count = 0
    for i in range(0, len(prediction)):


        if prediction[i] == test_list[i][6]:
            pos_count += 1
        else:
            neg_count += 1
    print('测试集总数', len(test_list))
    print('正确分类数', pos_count)
    print('错误分类数', neg_count)
    print('分类的准确率为', pos_count / len(test_list))
    # correct = [1 if a == b else 0 for a, b in zip(prediction, target)]
    # print(correct.count(1) / len(correct))  # 计算准确率


test()