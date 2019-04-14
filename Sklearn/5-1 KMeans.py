from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

data = np.random.random((200,2))
print(data)

k = 4

model = KMeans(n_clusters=k)
model.fit(data)

centers = model.cluster_centers_
results = model.predict(data)


x_min, x_max = data[:, 0].min() - 0.3, data[:, 0].max() + 0.3
y_min, y_max = data[:, 0].min() - 0.3, data[:, 0].max() + 0.3

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
cs = plt.contour(xx, yy, z)

marks = ['or', 'ob', 'og', 'oy']

for i,d in enumerate(data):
    plt.plot(d[0],d[1],marks[results[i]])

marks = ['*r', '*b', '*g', '*y']

for i, d in enumerate(centers):
    plt.plot(centers[0], centers[1], marks[results[i]])

plt.show()