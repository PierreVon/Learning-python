from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

data = np.random.random((200,2))

model = DBSCAN(eps=0.09, min_samples=4)
model.fit(data)

# we get some results
result = model.fit_predict(data)
print(result)

marks = ['or', 'ob', 'og', 'oy', 'ok', 'om']

# some data randomly distribute in whole graph in same color
# this kind of data can be seen as noise
for i,d in enumerate(data):
    plt.plot(d[0],d[1],marks[result[i]])

plt.show()