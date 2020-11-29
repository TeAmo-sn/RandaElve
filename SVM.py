# -*- coding:utf-8 -*-
from sklearn import svm
import numpy as np


def train_data():
    train_path = r'train_Feature.txt'
    data = np.loadtxt(train_path, dtype=float, delimiter=' ')
    np.set_printoptions(suppress=True)  # 不采用科学计数法
    # print(data)

    # 分离样本与标本，选取前5个,以行选取
    x, y = np.split(data, (5,), axis=1)

    clf_train = svm.SVC(C=10, kernel='poly', gamma=140, decision_function_shape='ovr')
    clf_train.fit(x, y.ravel())

    pre_path = r'data_feature.txt'
    pre_data = np.loadtxt(pre_path, dtype=float, delimiter=' ')
    np.set_printoptions(suppress=True)

    test_data, test_label = np.split(pre_data, (5,), axis=1)
    # print('精度：', clf_train.score(test_data, test_label))

    # 存储预测类别的数组
    predict_array = []
    predict_array = clf_train.predict(test_data)
    # print('测试分类标签：\n', predict_array)
    numb = 0
    for i in range(len(predict_array)):
        if predict_array[i] == -1:
            numb += 1
    print('numb:', numb)

    return clf_train


# train_data()
