# -*- coding : UTF-8-*-

import csv
import os
import struct
import time

import numpy as np
import cv2 as cv

import pyhht
from PyQt5.QtCore import QTimer
from _testcapi import INT_MIN, INT_MAX, CHAR_MIN, CHAR_MAX, LLONG_MIN, LLONG_MAX, SHRT_MAX, SHRT_MIN
from scipy import fftpack
from scipy.signal import argrelextrema
from PyQt5 import QtCore

import SVM
import processing

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi

INTERVAL = 0.0001
STRECHING = 5
COLUMN_SHIFT = 16
ROW_TWIST = 2018


class Randa_Project(QDialog):
    def __init__(self):
        super(Randa_Project, self).__init__()

        loadUi(r'RandaProcess.ui', self)

        self.dataMat = []
        self.processMat = []
        self.dztName = []

        # openfile
        self.loadDZTButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.loadDZTButton.clicked.connect(self.load_DZTFile)
        self.CurrentFile.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))

        # start_process
        self.processButton.clicked.connect(self.process_DZT)

        # save_image
        self.saveImageButton.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.saveImageButton.clicked.connect(self.save_Image)

        # save_process_image
        self.saveProImageButton.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.saveProImageButton.clicked.connect(self.save_ProImage)

        # Generate processing report
        # Define whether the processing button flag is clicked, and the report can only be generated after clicking
        self.flag = 0
        self.proReportButton.clicked.connect(self.proReport)

        # process_progress
        self.timer = QTimer(self.progressBar)
        self.progressBar.setValue(0)  # Initialize the progress bar
        self.progressBar.setMaximum(100)

    def load_DZTFile(self):
        self.dztName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files()")
        print(self.dztName)

        if len(self.dztName) != 0:
            # Display the currently loaded DZT path + name
            # self.cur_filename = self.dztName.split('/')[-1]
            self.loadDZTfile.setText(str(self.dztName))
            self.loadDZTfile.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))
            self.CurrentOP.setText(str('准备处理...'))
            self.CurrentOP.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))
            self.progressBar.setValue(0)

            self.dataArray, self.dzt_nsamp, self.dzt_interval, \
            self.dzt_lineCount, self.dzt_depth, self.dzt_range = load_dztFile(self.dztName)
            self.flag = 0

        else:
            QMessageBox.warning(self, '警告', 'Invalid File')

    def process_DZT(self):
        if len(self.dztName) != 0:
            self.progressBar.setValue(0)
            self.timer.timeout.connect(self.progress_change)
            self.timer.start(10)  # Call the progress_change function of the timeout connection every 1000ms

            # Load the original data file and get the data array, at this time dataArray is col * rh_nsamp
            # Process the original data file to obtain the characteristic value
            self.CurrentOP.setText(str('获取雷达数据特征值...'))
            self.CurrentOP.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))
            # Refresh the displayed text after setting (execute once after each setting)
            time.sleep(0.5)
            QApplication.processEvents()
            processing.process_data(self.dataArray.T)

            # Get training model
            self.CurrentOP.setText(str('开始训练特征数据...'))
            self.CurrentOP.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))
            time.sleep(0.5)
            QApplication.processEvents()
            clf_mod = SVM.train_data()

            # Prediction, get prediction label group
            self.CurrentOP.setText(str('预测数据分类...'))
            self.CurrentOP.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))
            time.sleep(0.5)
            QApplication.processEvents()
            T_flag = 1
            label_feature_path = r'data_feature.txt'
            self.labelArrayData = predict_data(label_feature_path, clf_mod)

            # Describe the processing effect through the data array and label array
            self.CurrentOP.setText(str('处理雷达数据...'))
            self.CurrentOP.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))
            time.sleep(0.5)
            QApplication.processEvents()
            self.dataMat, self.processMat, self.timeDepth, self.posLength = process_pic(self.dataArray,
                                                                                        self.labelArrayData,
                                                                                        self.dzt_nsamp,
                                                                                        self.dzt_interval,
                                                                                        self.dzt_lineCount,
                                                                                        self.dzt_depth,
                                                                                        self.dzt_range)

            self.flag = 1

        else:
            QMessageBox.warning(self, '警告', '未打开雷达文件')

    def progress_change(self):

        self.progressBar.setValue(self.progressBar.value() + 1)

        if self.progressBar.value() == 100:
            self.timer.stop()
            self.CurrentOP.setText(str('处理完成!'))
            time.sleep(0.2)
            QMessageBox.information(self, '提醒', '处理完成', QMessageBox.Yes, QMessageBox.Yes)
            self.CurrentOP.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))

    def save_Image(self):
        # Losslessly save as PNG format picture
        if len(self.dataMat) != 0:
            if self.flag == 1:
                save_Image_path, filetype = QFileDialog.getSaveFileName(self)
                if save_Image_path != '':
                    # picName = 'pic_image\\' + os.path.basename(self.pic_name).split('.')[0] + '.PNG'
                    cv.imwrite(save_Image_path, self.dataMat, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
                    QMessageBox.information(self, '提醒', '保存成功!', QMessageBox.Yes, QMessageBox.Yes)
                else:
                    QMessageBox.warning(self, '警告', '请选择保存位置!')
            else:
                QMessageBox.warning(self, '警告', '已加载新文件，尚未生成雷达图片!')
        else:
            QMessageBox.warning(self, '警告', '尚未生成雷达图片!')

    def save_ProImage(self):
        # Losslessly save as PNG format picture
        if len(self.processMat) != 0:
            if self.flag == 1:
                save_ProImage_path, filetype = QFileDialog.getSaveFileName(self)
                if save_ProImage_path != '':
                    # picName = 'pic_image\\' + os.path.basename(self.pic_name).split('.')[0] + '.PNG'
                    cv.imwrite(save_ProImage_path, self.processMat, [int(cv.IMWRITE_PNG_COMPRESSION), 0])
                    QMessageBox.information(self, '提醒', '保存成功!', QMessageBox.Yes, QMessageBox.Yes)
                else:
                    QMessageBox.warning(self, '警告', '请选择保存位置!')
            else:
                QMessageBox.warning(self, '警告', '已加载新文件，尚未生成雷达处理图片!')
        else:
            QMessageBox.warning(self, '警告', '尚未生成雷达处理图片!')

    def proReport(self):
        time_Start = []
        time_End = []
        pos_Start = []
        pos_End = []

        if self.flag == 1:
            report_name = (self.dztName.split('/')[-1]).split('.')[0] + 'Rep.csv'
            fo = open('dztReport//' + report_name, 'w', encoding='utf-8-sig', newline='')
            csv_writer = csv.writer(fo)
            if -1 not in self.labelArrayData:
                title = [u'雷达文件', u'病害区域时间开始位置(ns)', u'病害区域距离结束位置(m)']
                data = [self.dztName.split('/')[-1], 0, 0]
                csv_writer.writerow(title)
                csv_writer.writerow(data)

            else:
                title = [u'雷达文件', u'病害区域时间开始位置(ns)', u'病害区域时间结束位置(ns)', u'病害区域距离开始位置(m)', u'病害区域距离结束位置(m)']
                csv_writer.writerow(title)
                for i in range(len(self.timeDepth)):
                    if i % 2 == 0:
                        time_Start.append(self.timeDepth[i])
                        pos_Start.append(self.posLength[i])
                    else:
                        time_End.append(self.timeDepth[i])
                        pos_End.append(self.posLength[i])
                for j in range(len(time_Start)):
                    if j == 0:
                        data_report = [str(self.dztName.split('/')[-1]), float('%.4f' % time_Start[j]),
                                       float('%.4f' % time_End[j]),
                                       float('%.4f' % pos_Start[j]),
                                       float('%.4f' % pos_End[j])]
                        csv_writer.writerow(data_report)
                    else:
                        data_report = [' ', float('%.4f' % time_Start[j]), float('%.4f' % time_End[j]),
                                       float('%.4f' % pos_Start[j]),
                                       float('%.4f' % pos_End[j])]
                        csv_writer.writerow(data_report)

            # print('time_Start:', time_Start)
            # print('time_End:', time_End)
            # print('pos_Start:', pos_Start)
            # print('pos_End:', pos_End)
            QMessageBox.information(self, '提醒', '生成报告完成！', QMessageBox.Yes, QMessageBox.Yes)
        else:
            QMessageBox.information(self, '提醒', '请先进行雷达数据处理！', QMessageBox.Yes, QMessageBox.Yes)


# Forecast data
def predict_data(pre_path, pre_clf):
    pre_data = np.loadtxt(pre_path, dtype=float, delimiter=' ')
    np.set_printoptions(suppress=True)

    test_data, test_label = np.split(pre_data, (5,), axis=1)

    # Array of predicted categories
    predict_array = []
    predict_array = pre_clf.predict(test_data)
    # print('测试分类标签：\n', predict_array)

    return predict_array


# Load DZT file
def load_dztFile(dztfile):
    tmpString = []

    with open(dztfile, 'rb') as fp:

        print('File Name:', fp.name)

        fileSize = os.path.getsize(dztfile)  # Get file size
        print('fileSize: ', fileSize)

        fp.seek(2, 0)

        data1_1 = fp.read(2)
        rh_data = list(struct.unpack("H", data1_1))[0]
        data1_2 = fp.read(2)
        rh_nsamp = list(struct.unpack("H", data1_2))[0]
        data1_3 = fp.read(2)
        rh_bits = list(struct.unpack("H", data1_3))[0]
        print('rh_data:', rh_data)
        print('rh_nsamp:', rh_nsamp)
        print('rh_bits:', rh_bits)

        fp.seek(40, 0)
        data2 = fp.read(6)
        rh_rgain = list(struct.unpack("3H", data2))[0]
        rh_nrgain = list(struct.unpack("3H", data2))[1]
        rh_text = list(struct.unpack("3H", data2))[2]
        print('rh_rgain:', rh_rgain)
        print('rh_nrgain:', rh_nrgain)
        print('rh_text:', rh_text)

        fp.seek(10, 0)
        data3 = fp.read(20)
        rh_sps = list(struct.unpack("5f", data3))[0]
        rh_range = list(struct.unpack("5f", data3))[4]
        print('rh_sps', rh_sps)
        print('rh_range', rh_range)

        fp.seek(54, 0)
        data4 = fp.read(12)
        rh_depth = list(struct.unpack("3f", data4))[2]
        print('rh_depth', rh_depth)

        fp.seek(0, 0)

        if rh_bits == 8:
            dataBuffInt8 = 'a'
            min_size = CHAR_MIN
            max_size = CHAR_MAX
            print(dataBuffInt8, min_size, max_size)
        elif rh_bits == 16:
            dataBuffInt16 = 'b'
            min_size = SHRT_MIN
            max_size = SHRT_MAX
            print(dataBuffInt16, min_size, max_size)
        elif rh_bits == 32:
            min_size = INT_MIN
            max_size = INT_MAX
            # print(dataBuffInt32, min_size, max_size)
        elif rh_bits == 64:
            dataBuffInt64 = 'd'
            min_size = LLONG_MIN
            max_size = LLONG_MAX
            print(dataBuffInt64, min_size, max_size)

        lineCount = int((fileSize - rh_data + 1) / (rh_bits / 8 * rh_nsamp) + 0.5)
        print('total lines:', lineCount)

        fp.seek(rh_data + 4 * rh_nsamp * COLUMN_SHIFT, 0)
        print('data_start_pos:', fp.tell())
        nInsideCount = 0
        totalCount = rh_nsamp * lineCount
        interval = int(INTERVAL * max_size)
        print('spread range:', interval)

        # Read 2048 data each time, this is a signal, and then transpose to get 2048 rows of data
        dataSet = np.ones((lineCount - COLUMN_SHIFT, rh_nsamp))

        if rh_bits == 8:
            dzt_dataName = 'dztCSVData//' + (fp.name.split('/')[-1]).split('.')[0] + 'Data.csv'
            fo = open(dzt_dataName, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(fo)

            for row in range(lineCount - COLUMN_SHIFT):
                for j in range(rh_nsamp):
                    dataBuffInt8 = fp.read(4)
                    if len(dataBuffInt8) == 0:
                        break
                    else:
                        elem = struct.unpack('i', dataBuffInt8)[0]
                        # if np.math.isnan(elem):
                        # elem = 0.0
                        dataSet[row][j] = elem

                        if (dataSet[row][j] < interval) and (dataSet[row][j] > -1 * interval):
                            nInsideCount += 1

                        # Supplement the last 32 data
                        if row == (lineCount - COLUMN_SHIFT - 2) and j >= ROW_TWIST - 2:
                            dataSet[row + 1][j] = dataSet[row][j]
                            nInsideCount += 1

            # Splice the next thirty columns to the front
            supplement_min = dataSet[:, ROW_TWIST:rh_nsamp]
            supplement_max = dataSet[:, 0:ROW_TWIST]
            dataSet = np.concatenate((supplement_min, supplement_max), axis=1)

            # Write according to the column, generate 2048*col file
            for j in range(rh_nsamp):
                csv_writer.writerow(dataSet[:, j])

            print('dataSet.shape:', dataSet.shape)

        elif rh_bits == 16:
            dzt_dataName = 'dztCSVData//' + (fp.name.split('/')[-1]).split('.')[0] + 'Data.csv'
            fo = open(dzt_dataName, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(fo)

            for row in range(lineCount - COLUMN_SHIFT):
                for j in range(rh_nsamp):
                    dataBuffInt16 = fp.read(4)
                    if len(dataBuffInt16) == 0:
                        break
                    else:
                        elem = struct.unpack('i', dataBuffInt16)[0]
                        # if np.math.isnan(elem):
                        # elem = 0.0
                        dataSet[row][j] = elem

                        if (dataSet[row][j] < interval) and (dataSet[row][j] > -1 * interval):
                            nInsideCount += 1

                        # Supplement the last 32 data
                        if row == (lineCount - COLUMN_SHIFT - 2) and j >= ROW_TWIST - 2:
                            dataSet[row + 1][j] = dataSet[row][j]
                            nInsideCount += 1

            # Splice the next thirty columns to the front
            supplement_min = dataSet[:, ROW_TWIST:rh_nsamp]
            supplement_max = dataSet[:, 0:ROW_TWIST]
            dataSet = np.concatenate((supplement_min, supplement_max), axis=1)

            #  Write according to the column, generate 2048*col file
            for j in range(rh_nsamp):
                csv_writer.writerow(dataSet[:, j])

            print('dataSet.shape:', dataSet.shape)

        elif rh_bits == 32:
            dzt_dataName = 'dztCSVData//' + (fp.name.split('/')[-1]).split('.')[0] + 'Data.csv'
            fo = open(dzt_dataName, 'w', encoding='utf-8', newline='')
            # fo = open('dztCSVData//dzt_Data.csv', 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(fo)

            for row in range(lineCount - COLUMN_SHIFT):
                for j in range(rh_nsamp):
                    dataBuffInt32 = fp.read(4)
                    if len(dataBuffInt32) == 0:
                        break
                    else:
                        elem = struct.unpack('i', dataBuffInt32)[0]
                        # if np.math.isnan(elem):
                        # elem = 0.0
                        dataSet[row][j] = elem

                        if (dataSet[row][j] < interval) and (dataSet[row][j] > -1 * interval):
                            nInsideCount += 1

                        # Supplement the last 32 data
                        if row == (lineCount - COLUMN_SHIFT - 2) and j >= ROW_TWIST - 2:
                            dataSet[row + 1][j] = dataSet[row][j]
                            nInsideCount += 1

            # Splice the next thirty columns to the front
            supplement_min = dataSet[:, ROW_TWIST:rh_nsamp]
            supplement_max = dataSet[:, 0:ROW_TWIST]
            dataSet = np.concatenate((supplement_min, supplement_max), axis=1)

            #  Write according to the column, generate 2048*col file
            for j in range(rh_nsamp):
                csv_writer.writerow(dataSet[:, j])

            print('dataSet.shape:', dataSet.shape)

        elif rh_bits == 64:
            dzt_dataName = 'dztCSVData//' + (fp.name.split('/')[-1]).split('.')[0] + 'Data.csv'
            fo = open(dzt_dataName, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(fo)

            for row in range(lineCount - COLUMN_SHIFT):
                for j in range(rh_nsamp):
                    dataBuffInt64 = fp.read(4)
                    if len(dataBuffInt64) == 0:
                        break
                    else:
                        elem = struct.unpack('i', dataBuffInt64[0])
                        # if np.math.isnan(elem):
                        # elem = 0.0
                        dataSet[row][j] = elem

                        if (dataSet[row][j] < interval) and (dataSet[row][j] > -1 * interval):
                            nInsideCount += 1

                        # Supplement the last 32 data
                        if row == (lineCount - COLUMN_SHIFT - 2) and j >= ROW_TWIST - 2:
                            dataSet[row + 1][j] = dataSet[row][j]
                            nInsideCount += 1

            # Splice the next thirty columns to the front
            supplement_min = dataSet[:, ROW_TWIST:rh_nsamp]
            supplement_max = dataSet[:, 0:ROW_TWIST]
            dataSet = np.concatenate((supplement_min, supplement_max), axis=1)

            #  Write according to the column, generate 2048*col file
            for j in range(rh_nsamp):
                csv_writer.writerow(dataSet[:, j])

            print('dataSet.shape:', dataSet.shape)

        print(totalCount, nInsideCount, 'in the range')

    return dataSet, rh_nsamp, interval, lineCount, rh_depth, rh_range


# Processing radar data
def process_pic(processData, labelSet, rh_nsamp, interval, lineCount, rh_depth, rh_range):
    # For col row data, there are 2048 data in each row, and these 2048 data are sequentially transformed and stored
    # in dataMat
    dataMat = np.zeros((rh_nsamp, (lineCount - COLUMN_SHIFT) * STRECHING))
    for row in range(processData.shape[0]):
        for col in range(processData.shape[1]):
            indicator = (col + ROW_TWIST) % rh_nsamp

            if processData[row][indicator] < -1 * interval:
                for strech in range(STRECHING):
                    dataMat[col][row * STRECHING + strech] = 0
                continue
            if processData[row][indicator] > interval:
                for strech in range(STRECHING):
                    dataMat[col][row * STRECHING + strech] = 255
                continue
            currIndex = (processData[row][indicator] * 0.5 / interval + 0.5) * 255
            for strech in range(STRECHING):
                dataMat[col][row * STRECHING + strech] = currIndex

    # Now it is 2048 * col data
    processMat = dataMat.copy()

    # Storage time depth
    time_depth = []
    # Storage distance length
    pos_length = []
    # The coordinates of the upper left corner of the temporary storage frame
    pos_first = (0, 0)
    label_step = 1
    for row in range(processData.shape[0]):
        if labelSet[row] == -1:
            # Perform EMD decomposition
            decomposer = pyhht.emd.EMD(processData[row, :])

            # Get the IMF components after EMD decomposition
            imfs = decomposer.decompose()
            # IMF1 component
            reim_array = fftpack.hilbert(imfs[0])
            # Instantaneous amplitude or amplitude envelope
            Ins_amplitude = np.sqrt(imfs[0] ** 2 + reim_array ** 2)
            # print('Ins_amplitude:', Ins_amplitude)
            # Time depth of the maximum amplitude envelope
            IMF1_Max_Pos = np.argmax(Ins_amplitude, axis=0)
            # print('IMF1_Max_Pos:', IMF1_Max_Pos)

            # Get maximum value
            max_peaks = argrelextrema(processData[row, :], np.greater)
            max_peaks = max_peaks[0]

            # Get the upper and lower positions of the maximum time depth of the amplitude envelope
            # Conversion position, normalized to max_peaks, that is, look at the relative position of max_peaks
            trans_pos = int(float(IMF1_Max_Pos / rh_nsamp) * len(max_peaks))
            left_pos = trans_pos - 1
            right_pos = trans_pos + 1
            IMF1_Max_Pos = 0

            pos_up = (IMF1_Max_Pos + max_peaks[left_pos]) // 2

            # max_peaks[right_pos] is processed out of bounds, right_pos may be equal to the length of max_peaks
            if right_pos == len(max_peaks):
                right_pos = right_pos - 1

            pos_do = (IMF1_Max_Pos + max_peaks[right_pos]) // 2

            # Adjust two values, need pos_up <pos_do
            if pos_up > pos_do:
                pos_up, pos_do = pos_do, pos_up
            else:
                pos_up, pos_do = pos_up, pos_do

            first_point = (row * STRECHING, pos_up)
            if label_step == 1:
                if pos_up >= int(rh_nsamp / 3):
                    pos_up = 50
                    pos_first = (row * STRECHING, pos_up)
                else:
                    pos_first = first_point

            # Draw a rectangle for every thirty disease signals
            if label_step == 30:
                last_point = (row * STRECHING, pos_do)

                # save time depth from left_up to right_do
                time_depth.append(pos_first[1])
                time_depth.append(last_point[1])

                # save position length from left_up to right_do
                pos_length.append(pos_first[0])
                pos_length.append(last_point[0])

                label_step = 0

            label_step += 1

    print(time_depth)
    print(pos_length)
    # Draw rectangle
    green_line = (0, 255, 0)
    cut_time = np.array(time_depth.copy())
    cut_pos = np.array(pos_length.copy()) / 5
    # print(cut_time)
    # print(cut_pos)

    # Store the coordinate array of the start and end points of the rectangle for drawing the rectangle later
    row_array = []
    col_array = []

    # Standard deviation array, standard deviation mean, standard deviation of standard deviation,
    # distance between standard deviation and standard deviation mean
    data_std_array = []
    std_average = 0
    data_std_std = 0
    std_length = []
    # Choose a threshold for rectangle characterization
    for i in range(len(cut_time)):
        if i % 2 == 0:
            row_start = int(cut_time[i])
            row_end = int(cut_time[i + 1])
            col_start = int(cut_pos[i])
            col_end = int(cut_pos[i + 1])
            # print(row_start, row_end)
            # print(col_start, col_end)

            # Calculate the standard deviation of the data in the rectangular area
            data_std = np.std(processMat[row_start:row_end, col_start:col_end])
            # print('data_std:', data_std)

            data_std_array.append(data_std)

            # Get the coordinate array of the start and end points of the rectangle
            row_array.append(row_start)
            row_array.append(row_end)
            col_array.append(col_start * STRECHING)
            col_array.append(col_end * STRECHING)

            # print('row_array', row_array)
            # print('col_array', col_array)
            print('data_std_array', data_std_array)
        # Calculate the average and standard deviation of the standard deviation.
        # If the distance between the standard deviation and the average in the area
        # is greater than the standard deviation distance of all rectangular areas, remove the area

        if i == len(cut_time) - 2:
            std_average = np.average(data_std_array)
            data_std_std = np.std(data_std_array)
            std_length = np.abs(np.array(data_std_array) - std_average)
            # print('std_average', std_average)
            # print('data_std_std', data_std_std)
            # print('std_length', std_length)

    time_depth = []
    pos_length = []
    # Draw rectangles that meet the conditions
    for i in range(len(std_length)):
        if std_length[i] < data_std_std:
            cv.rectangle(processMat, (col_array[2 * i], row_array[2 * i]), (col_array[2 * i + 1], row_array[2 * i + 1]),
                         green_line, 2)
            time_depth.append(row_array[2 * i])
            time_depth.append(row_array[2 * i + 1])
            pos_length.append(col_array[2 * i])
            pos_length.append(col_array[2 * i + 1])
    # print('time2', time_depth)
    # print('pos2', pos_length)

    # save_ProImage_path = 'temp_ProImage\\' + 'temp_Image.PNG'
    # cv.imwrite(save_ProImage_path, processMat, [int(cv.IMWRITE_PNG_COMPRESSION), 0])

    # save_Image_path = 'temp_ProImage\\' + 'data_Image.PNG'
    # cv.imwrite(save_Image_path, dataMat, [int(cv.IMWRITE_PNG_COMPRESSION), 0])

    # Here the vertical position is converted into time depth, rh_range time total depth
    time_depth = np.array(time_depth) / rh_nsamp * rh_range
    # Here, the horizontal position is converted into distance length, rh_range time total depth
    pos_length = (np.array(pos_length) / 5) / lineCount * rh_depth * 2

    # print(time_depth)
    # print(pos_length)

    return dataMat, processMat, time_depth, pos_length


if __name__ == "__main__":
    # dzt_file = r'tmpData'
    #
    # for root, dirs, files in os.walk(dzt_file):
    #     for file in files:
    #         print('file name:', os.path.join(root, file))
    #
    #         # Load the original data file and get the data array, at this time dataArray is col * rh_nsamp
    #         dataArray, dzt_nsamp, dzt_interval, dzt_lineCount, dzt_depth, dzt_range = load_dztFile(
    #             os.path.join(root, file))
    #
    #         # Process the original data file to obtain the characteristic value
    #         processing.process_data(dataArray.T)
    #
    #         # Get training model
    #         clf = SVM.train_data()
    #
    #         # Prediction, get the predicted label group, now it is not expanded five times
    #         label_path = r'data_feature.txt'
    #         labelArray = predict_data(label_path, clf)
    #
    #         # Describe the processing effect through the data array and label array
    #         process_pic(dataArray, labelArray, dzt_nsamp, dzt_interval, dzt_lineCount, dzt_depth, dzt_range)

    # Adapt to high-resolution screens such as 2k, low-resolution screens can be defaulted
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    ui = Randa_Project()
    ui.show()

    sys.exit(app.exec_())
