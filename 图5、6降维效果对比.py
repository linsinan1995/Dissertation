import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import matplotlib as mpl
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import manifold
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.datasets.samples_generator import make_swiss_roll




# 图5



cov = np.diag([1,1,1])
mean = [0, 0, 0]
x1 = np.random.multivariate_normal(mean, cov, 500)
mean = [3, 3, 3]
x2 = np.random.multivariate_normal(mean, cov, 500)
mean = [6, 6, 6]
x3 = np.random.multivariate_normal(mean, cov, 500)
mean = [9, 9, 9]
x4 = np.random.multivariate_normal(mean, cov, 500)
mean = [12, 12, 12]
x5 = np.random.multivariate_normal(mean, cov, 500)
mean = [15, 15, 15]
x6 = np.random.multivariate_normal(mean, cov, 500)
x = np.concatenate((x1,x2,x3,x4,x5),axis=0)
y = np.zeros(2500)
y[0:500] = 1
y[500:1000] = 2
y[1000:1500] = 3
y[1500:2000] = 4
y[2000:2500] = 5

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(241, projection='3d')
ax.view_init(4, -72)
ax.scatter(x[:, 0], x[:, 1], x[:, 2],s=40,c=y, cmap=plt.cm.Spectral)

pca = PCA(n_components = 2).fit(x)
Y = pca.fit_transform(x)
ax = fig.add_subplot(242)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("PCA")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


lda = LinearDiscriminantAnalysis(n_components = 2).fit(x, y)
Y = lda.transform(x)
ax = fig.add_subplot(243)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("lda")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

pls = PLSRegression(n_components=2).fit(x, y)
Y = pls.transform(x)
ax = fig.add_subplot(244)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("%s" % "PLS")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

Y = manifold.MDS(n_components=2).fit_transform(x)
ax = fig.add_subplot(246)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("mds")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

Y = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(x)
ax = fig.add_subplot(247)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("Isomap")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

Y = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit_transform(x)
ax = fig.add_subplot(248)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("LLE")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()




# 图6  卷曲型数据




x, y= datasets.samples_generator.make_swiss_roll(n_samples=1500)
y[y<6] = 1
y[(y<8)&(y >= 6)] = 2
y[(y<10)&(y >= 8)] = 3
y[(y<12)&(y >= 10)] = 4
y[(y<15)&(y >= 12)] = 5

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(241, projection='3d')
ax.view_init(4, -72)
ax.scatter(x[:, 0], x[:, 1], x[:, 2],s=40,c=y, cmap=plt.cm.Spectral)

pca = PCA(n_components = 2).fit(x)
Y = pca.fit_transform(x)
ax = fig.add_subplot(242)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("PCA")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


lda = LinearDiscriminantAnalysis(n_components = 2).fit(x, y)
Y = lda.transform(x)
ax = fig.add_subplot(243)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("lda")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

pls = PLSRegression(n_components=2).fit(x, y)
Y = pls.transform(x)
ax = fig.add_subplot(244)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("%s" % "PLS")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

Y = manifold.MDS(n_components=2).fit_transform(x)
ax = fig.add_subplot(246)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("mds")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

Y = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(x)
ax = fig.add_subplot(247)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("Isomap")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

Y = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit_transform(x)
ax = fig.add_subplot(248)
plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("LLE")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()