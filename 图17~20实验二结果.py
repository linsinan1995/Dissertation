import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

dat = pd.read_csv('C:\Users\hasee\Desktop\data_ar.csv')

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax1.plot(dat.ix[:,0], dat.ix[:,1], "s-",
                label=u"费希尔判别法")
ax1.plot(dat.ix[:,0], dat.ix[:,2], "s-",
                label=u"主成分分析法")
ax1.plot(dat.ix[:,0], dat.ix[:,3], "s-",
                label=u"多维尺度分析法")
ax1.plot(dat.ix[:,0], dat.ix[:,4], "s-",
                label=u"等度量映射法")
ax1.plot(dat.ix[:,0], dat.ix[:,5], "s-",
                label=u"偏最小二乘法")
ax1.plot(dat.ix[:,0], dat.ix[:,6], "s-",
                label=u"局部线性嵌入法")

base = [0.4] * 50     # 朴素贝叶斯baseline
base1 = [0.1] * 50    # SVM
ax1.plot(np.linspace(0,160,50), base, "--",color='gray',label=u"未降维时")

ax1.plot(np.linspace(0,160,50), base1, "-",color='black',
                label=u"高斯核的SVM")
ax1.set_ylabel(u"分类错误率",fontsize="13.5")
ax1.set_ylim([0.0, 1.05])
ax1.legend(loc="upper right",fontsize="15")
ax1.set_title(u'IMM数据集中降维方法对朴素贝叶斯方法的分类表现影响',fontsize="14")

plt.tight_layout()
plt.show()