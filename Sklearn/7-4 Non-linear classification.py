import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn import tree

x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5)

model = tree.DecisionTreeClassifier(max_depth=4)
model.fit(x_train, y_train)

print(classification_report(y_train, model.predict(x_train)))
print(classification_report(y_test, model.predict(x_test)))

x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z)

plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.show()