#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: Linsinan

from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import random
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import gzip
import pandas as pd
from PIL import Image
from sklearn import linear_model
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import FactorAnalysis
from sklearn import manifold
from sklearn.decomposition import IncrementalPCA
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.naive_bayes import GaussianNB
import random
from sklearn.model_selection import KFold
from PIL import Image


def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data

def Kfold_data(train_index, test_index, x, y):
    train_x = x[train_index]
    train_y = y[train_index]
    test_x = x[test_index]
    test_y = y[test_index]

    return train_x, train_y, test_x, test_y

def load_data(type=0,train_kfold=0):
    #0：yale,1:ar,2:orl,3:olivetti,4:imm
    if int(type) == 0:
        for j in range(15):
            for i in range(11):
                filename = 'C:\\Users\\hasee\\Desktop\\yale\\%d\\s%d.bmp' % (j + 1, i + 1)
                # print filename
                if i == 0 and j == 0:
                    x = ImageToMatrix(filename).ravel()
                else:
                    x = np.concatenate((x, ImageToMatrix(filename).ravel()), axis=0)
        y = np.ones(165)
        for i in range(15):
            for j in range(11):
                y[i * 11 + j] = i + 1

    if int(type) == 1:
        import os
        file = os.listdir('C:\\Users\\hasee\\Desktop\\AR\\')
        k = 0
        for i in file:
            k += 1
            filename = 'C:\\Users\\hasee\\Desktop\\AR\\' + i
            if k == 1:
                x = ImageToMatrix(filename).ravel()
            else:
                x = np.concatenate((x, ImageToMatrix(filename).ravel()), axis=0)

        y = np.ones(1680)
        for i in range(120):
            for j in range(14):
                y[i * 14 + j] = i + 1

    if int(type) == 2:
        import os
        file = os.listdir('C:\\Users\\hasee\\Desktop\\ORL\\')
        k = 0
        for i in file:
            filename = 'C:\\Users\\hasee\\Desktop\\ORL\\' + i
            file2 = os.listdir(filename)
            for j in file2:
                if j.split(".")[1] == 'bmp':
                    k += 1
                    filename3 = filename + "\\" + j
                    if k == 1:
                        x = ImageToMatrix(filename3).ravel()
                    else:
                        x = np.concatenate((x, ImageToMatrix(filename3).ravel()), axis=0)
                else:
                    pass
        y = np.zeros(400)
        for i in range(40):
            for j in range(10):
                y[i * 10 + j] = i + 1
    if int(type) == 3:
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces()
        x = data.images.reshape((len(data.images), -1))
        y = data.target
    if int(type) == 4:
        import os
        k = 0
        for i in range(240):
            filename = 'C:\\Users\\hasee\\Desktop\\crop2\\%d.jpg' % i
            k += 1
            if k == 1:
                x = ImageToMatrix(filename).ravel()
            else:
                x = np.concatenate((x, ImageToMatrix(filename).ravel()), axis=0)
        y = np.zeros(240)
        for i in range(40):
            for j in range(6):
                y[i * 6 + j] = i + 1
    if int(train_kfold) <=0 :
        index = np.array(random.sample(range(len(y)), len(y)))
        x = x[index]
        y = y[index]
        return x, y
    else:
        index = np.array(random.sample(range(len(y)), len(y)))
        test_index = index[0:round(len(y) / int(train_kfold))]
        train_index = index[round(len(y) / int(train_kfold))::]
        train_x = x[train_index]
        train_y = y[train_index]
        test_x = x[test_index]
        test_y = y[test_index]
        return train_x, train_y,test_x,test_y

def test_clf(clf,type=0,times=10):
    print u'分类器：', clf
    x,y= load_data(type)
    kf = KFold(n_splits=times)
    index = np.array(random.sample(range(len(y)), len(y)))
    scores = np.zeros(times)
    train_times = np.zeros(times)
    datasets = ['Yale','AR','orl','oli','IMM']
    count = 0
    print u'%d折交叉检验开始，数据加载中，数据集为%s：' % (times, datasets[type])
    for train_index, test_index in kf.split(index):
        count += 1
        print u'\n==========第%d次训练开始==========\n' % count
        train_x, train_y, test_x, test_y = Kfold_data(train_index, test_index, x, y)

        if hasattr(clf, 'alpha'):
            alpha_can = np.logspace(-3, 2, 10)
            model = GridSearchCV(clf, param_grid={'alpha': alpha_can}, cv=5)
            model.set_params(param_grid={'alpha': alpha_can})
            m = alpha_can.size
        else:
            model = clf
        if hasattr(clf, 'n_neighbors'):
            neighbors_can = np.arange(5, 20)
            model = GridSearchCV(clf, param_grid={'n_neighbors': neighbors_can}, cv=5)
            model.set_params(param_grid={'n_neighbors': neighbors_can})
            m = neighbors_can.size
        if hasattr(clf, 'C'):
            C_can = np.logspace(1, 3, 3)
            gamma_can = np.logspace(-3, 0, 3)
            model = GridSearchCV(clf, param_grid={'C':C_can, 'gamma':gamma_can}, cv=5)
            model.set_params(param_grid={'C':C_can , 'gamma':gamma_can})
            m = C_can.size # * gamma_can.size
        if hasattr(clf, 'max_depth'):
            max_depth_can = np.arange(5, 20)
            model = GridSearchCV(clf, param_grid={'max_depth': max_depth_can}, cv=5)
            model.set_params(param_grid={'max_depth': max_depth_can})
            m = max_depth_can.size
        t_start = time()
        model.fit(train_x, train_y)
        t_end = time()
        if str(clf).split('(')[0] != 'GaussianNB':
            t_train = (t_end - t_start)/m
            print u'基于灰度搜索5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒' % ((t_end - t_start), m, t_train)
            print u'最优超参数为：', model.best_params_
        else:
            t_train = t_end - t_start
            print u'训练时间为：%.3f秒' % t_train
        y_hat = model.predict(test_x)
        acc = metrics.accuracy_score(test_y, y_hat)
        scores[count-1] = acc
        train_times[count-1] = t_train
        print u'测试集准确率：%.2f%%' % (100 * acc)

    name = str(clf).split('(')[0]
    index = name.find('Classifier')
    if index != -1:
        name = name[:index]     # 去掉末尾的Classifier
    if name == 'SVC':
        name = 'SVM'
    return train_times.mean(), 1-scores.mean(), name


if __name__ == "__main__":


    print u'\n\n======================================\n分类器的比较：\n'
    clfs = (GaussianNB(),
            KNeighborsClassifier(),         # 19.737(0.282), 0.208, 86.03%
            DecisionTreeClassifier(),              # 25.6(0.512), 0.003, 89.73%
            RandomForestClassifier(n_estimators=50),   # 59.319(1.977), 0.248, 77.01%
            SVC()                 #kernel='linear'          # 236.59(5.258), 1.574, 90.10%
            )
    result = []
    # 选择数据集
    data_type = 4
    for clf in clfs:
        a = test_clf(clf,type=data_type)
        result.append(a)
        print '\n'
    result = np.array(result)
    time_train, err, names = result.T
    names = ['NB', 'KNN', 'CART', 'RF', 'SVM']
    x = np.arange(len(time_train))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(5, 5), facecolor='w')
    ax = plt.axes()
    plt.ylim([0,1])
    b1 = ax.bar(x, err, width=0.25, color='#77E0A0')
    ax_t = ax.twinx()
    b2 = ax_t.bar(x+0.25, time_train, width=0.25, color='#FFA0A0')
    plt.xticks(x+0.125, names, fontsize=18)
    leg = plt.legend([b1[0], b2[0]], (u'错误率', u'训练时间'), loc='upper left', shadow=True,fontsize='x-large')
    # for lt in leg.get_texts():
    #     lt.set_fontsize(14)
    datasets = ['Yale', 'AR', 'orl','oli', 'IMM']
    _title =  datasets[data_type] + u'数据集不同分类器间的比较'
    plt.title(_title, fontsize=18)
    plt.xlabel(u'分类器名称')
    plt.grid(True)
    plt.tight_layout(2)
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(18)
    plt.show()

# debug
# xx=x[68]
# xx.shape=[200,200]
# MatrixToImage(xx).show()