# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:03:04 2017
@author: Administrator
"""

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

# 数据读取
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
    
    


# 降维实验
types = 4
x,y = load_data(type=types)
model = GaussianNB()
# model = LogisticRegression(penalty='l2')
times = 10
scores = 1 - cross_val_score(model, x, y, cv=times)

# model = LogisticRegression()
print "未降维分类准确率： %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
l = [1, 3,  5,  7, 10, 13, 15, 17, 20 ,23 , 25, 27, 30, 33, 35, 37, 40, 43, 45, 47, 50, 53, 55, 57, 60, 63, 65, 67, 70, 73, 75, 77, 80]#,82,85,87,90,92,95,97,100,102,105,110,115,120,125,130,135,140]
lda_score = np.zeros(len(l))
pca_score = np.zeros(len(l))
MDS_score = np.zeros(len(l))
isomap_score = np.zeros(len(l))
pls_score = np.zeros(len(l))
LLE_score = np.zeros(len(l))

lda_std = np.zeros(len(l))
pca_std = np.zeros(len(l))
MDS_std = np.zeros(len(l))
isomap_std = np.zeros(len(l))
pls_std = np.zeros(len(l))
LLE_std = np.zeros(len(l))


count = 0
for i in l:
    # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1.0/times)


    # pca
    pca = PCA(n_components=i).fit(x)
    x2 = pca.transform(x)
    cv_pca = 1 - cross_val_score(model, x2, y, cv=times)
    pca_score[count] = cv_pca.mean()
    pca_std[count] = cv_pca.std()

    # MDS
    mds = manifold.MDS(n_components=i, eps=1e-10)
    x2 = mds.fit(x).embedding_
    cv_MDS = 1 - cross_val_score(model, x2, y, cv=times)
    MDS_score[count] = cv_MDS.mean()
    MDS_std[count] = cv_MDS.std()

    # Isomap
    x2 = manifold.Isomap(60, n_components=i).fit_transform(x)
    cv_Isomap = 1 - cross_val_score(model, x2, y, cv=times)
    isomap_score[count] = cv_Isomap.mean()
    isomap_std[count] = cv_Isomap.std()


    # LLE
    x2 = LocallyLinearEmbedding(n_neighbors=60, n_components=i).fit_transform(x)
    cv_LLE = 1 - cross_val_score(model, x2, y, cv=times)
    LLE_score[count] = cv_LLE.mean()
    LLE_std[count] = cv_LLE.std()

    # lda
    kf = KFold(n_splits=times)
    cv_lda = np.zeros(times)
    count2 = 0
    index = np.array(random.sample(range(len(y)), len(y)))
    for train, test in kf.split(index):
        index_train = index[train]
        index_test = index[test]
        lda = LinearDiscriminantAnalysis(n_components=i).fit(x[index_train], y[index_train])
        x2 = lda.transform(x[index_train])
        x3 = lda.transform(x[index_test])
        model.fit(x2, y[index_train])
        predict = model.predict(x3)
        accuracy_lda = metrics.accuracy_score(y[index_test], predict)
        cv_lda[count2] = 1 - accuracy_lda
        count2 += 1
    lda_score[count] = cv_lda.mean()
    lda_std[count] = cv_lda.std()
    # pls
    cv_pls = np.zeros(times)
    count2 = 0
    for train, test in kf.split(index):
        index_train = index[train]
        index_test = index[test]
        pls = PLSRegression(n_components=i).fit(x[index_train], y[index_train])
        x2 = pls.transform(x[index_train])
        x3 = pls.transform(x[index_test])
        model.fit(x2, y[index_train])
        predict = model.predict(x3)
        accuracy_pls = metrics.accuracy_score(y[index_test], predict)
        cv_pls[count2] = 1 - accuracy_pls
        count2 += 1
    pls_score[count] = cv_pls.mean()
    pls_std[count] = cv_pls.std()


    print '维度为%d' % i
    print 'pca降维后分类错误率： %0.2f (+/- %0.2f)' % (cv_pca.mean(), cv_pca.std() * 2)
    print 'mds降维后分类错误率： %0.2f (+/- %0.2f)' % (cv_MDS.mean(), cv_MDS.std() * 2)
    print 'Isomap降维后分类错误确率： %0.2f (+/- %0.2f)' % (cv_Isomap.mean(), cv_Isomap.std() * 2)
    print 'lda降维后分类错误率： %0.2f (+/- %0.2f)' % (cv_lda.mean(), cv_lda.std() * 2)
    print 'pls降维后分类错误率： %0.2f (+/- %0.2f)' % (cv_pls.mean(), cv_pls.std() * 2)
    print 'LLE降维后分类错误率： %0.2f (+/- %0.2f)' % (cv_LLE.mean(), cv_LLE.std() * 2)

    count += 1
    
# 第一版的绘图
def pplot(score = lda_score,std = lda_std,l=l,Name='LDA'):

    plt.figure().set_size_inches(16, 12)
    plt.plot(l,score)
    plt.ylim(0,1)
    up = score + 2*std
    down = score - 2*std
    up[up >= 0.999] = 0.998
    down[down <= 0] = 0.001

    # plot error lines showing +/- std. errors of the scores
    plt.plot(l, up, 'b--')
    plt.plot(l, down, 'b--')

    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(l, up ,down, alpha=0.15)

    plt.ylabel('accuracy +/- std error')
    plt.xlabel('dimensions')
    plt.title(Name)
    plt.axhline(scores.mean(), linestyle='--', color='.5')

pplot()

pplot(score = pca_score,std = pca_std,l=l,Name='pca')
pplot(score = MDS_score,std = MDS_std,l=l,Name='MDS')
pplot(score = isomap_score,std = isomap_std,l=l,Name='isomap')
pplot(score = pls_score,std = pls_std,l=l,Name='pls')
pplot(score = LLE_score,std = LLE_std,l=l,Name='LLE')
    


#  数据储存
import pandas as pd
a=pd.DataFrame([lda_score,pca_score,MDS_score,isomap_score,pls_score,LLE_score,lda_std,pca_std,MDS_std,isomap_std,pls_std,LLE_std]).T
a.columns = ['lda','pca','mds','isomap','pls','lle','slda','spca','smds','sisomap','spls','slle']
a.index = l
a.to_csv(r'C:\Users\hasee\Desktop\rotate_yale.csv')
    
    
# rotate
# NB 0.21 (+/- 0.16)
#