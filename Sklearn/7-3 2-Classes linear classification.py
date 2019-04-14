from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.metrics import classification_report

x_data, y_data = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

model = tree.DecisionTreeClassifier()
model.fit(x_data, y_data)

print(classification_report(y_data, model.predict(x_data)))

x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z)

plt.scatter(x_data[:,0], x_data[:,1], c= y_data)
plt.show()