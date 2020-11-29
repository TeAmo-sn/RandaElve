# -*- coding:utf-8 -*-
import csv
import numpy as np
import math

from matplotlib.ticker import MultipleLocator
from pyhht import EMD
from scipy import fftpack
from scipy.signal import hilbert
import numpy.fft as fft
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_true_data(array_data):
    true_value = []
    for rows in range(array_data.shape[0]):
        for cols in range(20):
            if array_data[rows][cols] != '':
                true_value.append(float(array_data[rows][cols]))  # 转string 为 float
            else:
                true_value.append(0.0)
    # 不使用科学计数法表示
    np.set_printoptions(suppress=True)
    true_value = np.array(true_value)
    true_value = true_value.reshape((2048, 20))
    # print(true_value)

    return true_value


def get_false_data(array_data):
    false_value = []
    for rows in range(array_data.shape[0]):
        for cols in range(28, 48):
            if array_data[rows][cols] != '':
                false_value.append(float(array_data[rows][cols]))  # 转string 为 float
            else:
                false_value.append(0.0)

    # 不使用科学计数法表示
    np.set_printoptions(suppress=True)
    false_value = np.array(false_value)
    false_value = false_value.reshape((2048, 20))
    # print(false_value)

    return false_value


def get_max_value(array):
    # 求出最大值
    max_value = np.max(array, axis=0)
    max_value = (max_value - np.min(array, axis=0)) / np.var(array, axis=0, ddof=1) * 10000
    # print(max_value)
    return max_value


def get_max_value_pos(array):
    # 求出最大值的位置
    max_pos = np.argmax(array, axis=0) / 2048
    max_pos = (max_pos - np.argmin(array, axis=0)) / (np.argmax(array, axis=0) - np.argmin(array, axis=0))

    # max_pos = np.around(max_pos, decimals=3)  # 保留小数点后三位
    # print('-------正常区域最大值位置------')
    # print(max_pos)
    return max_pos


def get_ene_value(array):
    # 求出反射信号能量值
    square_value = np.square(array / np.std(array, axis=0))  # 归一化后平方
    ene_value = np.sum(square_value, axis=0) / 10000  # 求和

    # ene_value = np.around(ene_value, decimals=3)  # 设置范围小数点后保留三位
    # print('-------正常区域反射信号能量值------')

    # print(ene_value)
    return ene_value


def get_Spectral_entropy(array):
    # 获得频谱熵
    # 获得离散的频谱序列

    array_mean = np.mean(array, axis=0)
    # print('mean值: \n', array_mean)
    array = array - array_mean  # 去掉其直流的部分, 使其频谱可以更准确地表现出信号的能量大小。

    sequence_value = fft.fft(array)
    # print('fft值:\n', sequence_value)

    # 获得一维数组前半序列
    N = math.ceil(sequence_value.shape[0] / 2)
    half_sequence = []  # 前半序列

    for rows in range(N):
        for cols in range(sequence_value.shape[1]):
            half_sequence.append(sequence_value[rows][cols])

    half_sequence = np.array(half_sequence)
    half_sequence = half_sequence.reshape((N, sequence_value.shape[1]))
    # print('half_sequence:', half_sequence)

    # 计算概率
    pk = []  # 概率
    cols_value = []  # 暂存每一列的值
    Hm = []  # 频谱熵
    for col in range(sequence_value.shape[1]):
        cols_value = half_sequence[:, col]
        pk = (abs(cols_value) ** 2 / sum(abs(cols_value) ** 2))

        # 去除PK中为0的值，避免出现np.log(pk)不存在
        pk1 = []
        for i in range(N):
            if pk[i] != 0:
                pk1.append(pk[i])

        Hm.append(- sum(pk1 * np.log(pk1)))

    Hm = Hm / np.log(N)  # 谱熵归一化 效果并不明显
    # Hm = (Hm - np.mean(Hm)) * 100

    # Hm = np.around(Hm, decimals=3)  # 设置范围小数点后保留三位

    # print('频谱熵值: \n', Hm)

    return Hm


def get_Time_Spectral_entropy(array):
    # 获得时频谱分量熵
    # 获得离散的频谱序列
    # print(array)

    array_mean = np.mean(array, axis=0)
    # print('mean值: \n', array_mean)
    array = array - array_mean  # 去掉其直流的部分, 使其频谱可以更准确地表现出信号的能量大小。

    # 获得一维数组前半序列

    N = math.ceil(array.shape[0] / 2)
    half_sequence = []  # 前半序列

    for rows in range(N):
        for cols in range(array.shape[1]):
            half_sequence.append(array[rows][cols])

    half_sequence = np.array(half_sequence)
    half_sequence = half_sequence.reshape((N, array.shape[1]))

    # 计算概率
    pk = []  # 概率

    cols_value = []  # 暂存每一列的值
    Sh = []  # 频谱熵
    for col in range(array.shape[1]):
        cols_value = half_sequence[:, col]
        pk = (abs(cols_value) ** 2 / sum(abs(cols_value) ** 2))

        # 去除PK中为0的值，避免出现np.log(pk)不存在
        pk1 = []
        for i in range(N):
            if pk[i] != 0:
                pk1.append(pk[i])

        Sh.append(- sum(pk1 * np.log(pk1)))

    # Sh = Sh / np.log(N)  # 谱熵归一化

    # Sh = np.around(Sh, decimals=3)  # 设置范围小数点后保留三位

    # print('时频谱熵值: \n', Sh)

    return Sh


def get_Max_amplitude(array):
    # 最大振幅数组
    max_amplitude = []
    for col in range(array.shape[1]):
        DataRaw = array[:, col]
        # 进行EMD分解
        decomposer = EMD(DataRaw)
        # print('decomposer:', decomposer)

        # 获取EMD分解后的IMF成分
        imfs = decomposer.decompose()
        # print('imfs:', len(imfs))
        IMF1_Max_Pos = np.argmax(imfs[0])
        # print('IMF1_Max_Pos:', IMF1_Max_Pos)

        # 分解后的组分数
        n_components = imfs.shape[0]

        # 瞬时振幅数组
        Ins_amplitude = []
        # 当前分量最大振幅数组
        Cur_max_amplitude = []
        for i in range(n_components):
            # 希尔伯特变换数组
            reim_array = []
            reim_array = hilbert(imfs[i])

            # 瞬时振幅
            # Ins_amplitude = np.sqrt(np.square(imfs[i]) + np.square(reim_array))
            Ins_amplitude = abs(reim_array)
            Cur_max_amplitude.append(np.max(Ins_amplitude))

        max_amplitude.append(np.max(Cur_max_amplitude))

    # max_amplitude = (max_amplitude - np.mean(max_amplitude)) / np.std(max_amplitude)
    # max_amplitude = (max_amplitude - np.min(max_amplitude)) / (np.max(max_amplitude) - np.min(max_amplitude))
    max_amplitude = preprocessing.scale(max_amplitude)
    # print('max_amplitude: \n', max_amplitude)

    return max_amplitude


def get_Max_amplitude1(array):
    # 最大振幅数组
    max_amplitude = []
    for col in range(array.shape[1]):
        DataRaw = array[:, col]
        # 进行EMD分解
        decomposer = EMD(DataRaw)
        # print('decomposer:', decomposer)

        # 获取EMD分解后的IMF成分
        imfs = decomposer.decompose()

        # 瞬时振幅数组
        Ins_amplitude = []

        # IMF1分量
        reim_array = fftpack.hilbert(imfs[0])

        # 瞬时振幅 或者称为包络
        Ins_amplitude = np.sqrt(imfs[0] ** 2 + reim_array ** 2)
        # Ins_amplitude = abs(reim_array)
        max_amplitude.append(np.max(Ins_amplitude))
        # print('max_amplitude\n', max_amplitude)

    # 沿着某个轴标准化数据集，以均值为中心，以分量为单位方差。
    max_amplitude = (max_amplitude - np.min(max_amplitude)) / (np.max(max_amplitude) - np.min(max_amplitude))
    # max_amplitude = preprocessing.scale(max_amplitude)
    # print('max_amplitude: \n', max_amplitude)

    return max_amplitude


def plotResult(matrix1, matrix2):
    x = np.arange(len(matrix1))
    ax = plt.subplot()
    ax.scatter(x, matrix1, alpha=0.5, label="nor")
    ax.scatter(x, matrix2, c='green', alpha=0.6, label="abnor")  # 改变颜色 病害信号
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-0.5, len(matrix1))
    plt.show()


def processing_data(filename, savepath):
    # 提取特征值
    # 分别存储每个文件正常区域特征值和病害区域特征值
    normal_value = []
    abnormal_value = []
    # 打开文件：
    train_data = []

    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            train_data.append(row)

    array_data = np.array(train_data)

    # 获取非病害区域特征值
    true_data = get_true_data(array_data)

    # print('-------正常区域最大振幅幅值------')
    true_max_value = get_max_value(true_data)
    normal_value.append(true_max_value)

    # print('-------正常区域最大振幅位置------')
    true_max_pos = get_max_value_pos(true_data)
    normal_value.append(true_max_pos)

    # print('-------正常区域信号能量------')
    true_ene_value = get_ene_value(true_data)
    normal_value.append(true_ene_value)

    # print('-------正常区域频谱熵值------')
    # true_Spectral_entropy = get_Spectral_entropy(true_data)
    # normal_value.append(true_Spectral_entropy)

    # print('-------正常区域时频谱熵值------')
    true_Time_Spectral_entropy = get_Time_Spectral_entropy(true_data)
    normal_value.append(true_Time_Spectral_entropy)

    # print('-------正常区域IMF1分量最大振幅------')
    true_max_amplitude = get_Max_amplitude1(true_data)
    normal_value.append(true_max_amplitude)

    normal_value = np.array(normal_value)
    normal_value = normal_value.T
    # print(normal_value)

    true_flag = np.ones(20)
    normal_value = np.column_stack((normal_value, true_flag))
    # print(normal_value)

    # 获取病害区域特征值
    false_data = get_false_data(array_data)

    # print('-------病变区域最大振幅幅值------')
    false_max_value = get_max_value(false_data)
    abnormal_value.append(false_max_value)

    # print('-------病变区域最大振幅位置------')
    false_max_pos = get_max_value_pos(false_data)
    abnormal_value.append(false_max_pos)

    # print('-------病变区域信号能量------')
    false_ene_value = get_ene_value(false_data)
    abnormal_value.append(false_ene_value)

    # print('-------病害区域频谱熵值------')
    # false_Spectral_entropy = get_Spectral_entropy(false_data)
    # abnormal_value.append(false_Spectral_entropy)

    # print('-------病害区域时频谱熵值------')
    false_Time_Spectral_entropy = get_Time_Spectral_entropy(false_data)
    abnormal_value.append(false_Time_Spectral_entropy)

    # print('-------病变区域IMF1分量最大振幅------')
    false_max_amplitude = get_Max_amplitude1(false_data)
    abnormal_value.append(false_max_amplitude)

    abnormal_value = np.array(abnormal_value)
    abnormal_value = abnormal_value.T

    false_flag = np.zeros(20) - 1
    abnormal_value = np.column_stack((abnormal_value, false_flag))
    # print(abnormal_value)

    # 写入到txt文件中
    with open(savepath, 'ab') as file:
        np.savetxt(file, normal_value, fmt="%.4f %.4f %.4f %.4f %.4f %.f")

    with open(savepath, 'ab') as file:
        np.savetxt(file, abnormal_value, fmt="%.4f %.4f %.4f %.4f %.4f %.f")

    # 绘图画出每组特征值位置
    # plotResult(true_max_value, false_max_value)  # 尚可
    # plotResult(true_max_pos, false_max_pos)  # 犬牙交错 -> 区别开（方式不太行）
    # plotResult(true_ene_value, false_ene_value)  # 泾渭分明
    # plotResult(true_Spectral_entropy, false_Spectral_entropy)  # 唇亡齿寒
    # plotResult(true_Time_Spectral_entropy, false_Time_Spectral_entropy)  # 老死不相往来
    # plotResult(true_max_amplitude, false_max_amplitude)  # 融汇贯通


def get_data(filename):
    # 获得DZT文件的特征值
    # 打开文件：
    dzt_data = []

    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            dzt_data.append(row)

    array_data = np.array(dzt_data)

    # print('array_data\n', array_data)
    # print(array_data.shape)

    data_value = []
    for rows in range(array_data.shape[0]):
        for cols in range(array_data.shape[1]):
            if array_data[rows][cols] != '':
                data_value.append(float(array_data[rows][cols]))  # 转string 为 float
            else:
                data_value.append(0.0)
    # 不使用科学计数法表示
    np.set_printoptions(suppress=True)
    data_value = np.array(data_value)
    data_value = data_value.reshape(2048, array_data.shape[1])

    # print('data_value\n', data_value)
    # print(data_value.shape)

    return data_value


def process_data(totaldata):
    # 总数据的处理
    value_array = []

    # print('-------最大振幅幅值------')
    dzt_max_value = get_max_value(totaldata)
    value_array.append(dzt_max_value)

    # print('-------最大振幅位置------')
    dzt_max_pos = get_max_value_pos(totaldata)
    value_array.append(dzt_max_pos)

    # print('-------信号能量------')
    dzt_ene_value = get_ene_value(totaldata)
    value_array.append(dzt_ene_value)

    # print('-------频谱熵值------')
    # dzt_Spectral_entropy = get_Spectral_entropy(totaldata)
    # value_array.append(dzt_Spectral_entropy)

    # print('-------时频谱熵值------')
    dzt_Time_Spectral_entropy = get_Time_Spectral_entropy(totaldata)
    value_array.append(dzt_Time_Spectral_entropy)

    # print('-------IMF1分量最大振幅------')
    dzt_max_amplitude = get_Max_amplitude1(totaldata)
    value_array.append(dzt_max_amplitude)

    value_array = np.array(value_array)
    value_array = value_array.T
    # print(value_array)

    # 写入到txt文件中
    data_path = "data_feature.txt"
    with open(data_path, 'wb') as file:
        np.savetxt(file, value_array, fmt="%.4f %.4f %.4f %.4f %.4f")


def read_file():
    # 读取训练特征值地址
    file_train = r'trainData'
    file_test = r'testData'
    data_test = r'test.csv'

    # 保存文件地址
    train_path = "train_Feature.txt"
    test_path = 'test_dataIMF1.txt'
    train_data_test = 'train_data_test.txt'

    # 新建列表，存放文件名
    filename_csv = []

    # flag = input('a:train  b:test\n')

    # 循环读取文件夹下的文件名
    for root, dirs, files in os.walk(file_train):
        for file in files:
            filename_csv.append(os.path.join(root, file))
            # processing_data(os.path.join(root, file), train_path)

    # 循环读取文件夹下的文件名
    for root, dirs, files in os.walk(file_test):
        for file in files:
            filename_csv.append(os.path.join(root, file))
            # processing_data(os.path.join(root, file), train_path)

    # 获得原数据文件特征值
    # data_value = get_data(data_test)
    # process_data(data_value)

    # processing_data(data_test, train_data_test)


# read_file()
