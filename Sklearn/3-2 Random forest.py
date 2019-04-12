from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt

x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
# divide data set into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5)


def plot(model):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z)


dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)
print("Decision tree's accuracy is: ", dtree.score(x_test, y_test))

# Random Forest
# 1. Using bagging Randomly extracts m subsets from data set
# 2. Randomly choose k attributes from attribute set (n>k), building tree by best split node from k attributes
# 3. Creating m CART through 1 and 2
# 4. Voting to choose maximum number of class
RF = RandomForestClassifier(n_estimators=50)
RF.fit(x_train, y_train)
print("RF's accuracy is: ", RF.score(x_test, y_test))

plt.subplot(121)
plot(dtree)
plt.title('Decision Tree')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.subplot(122)
plot(RF)
plt.title('Random Forest')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.show()