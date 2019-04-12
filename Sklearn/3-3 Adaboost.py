from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

x1, y1 = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
# mean = (3, 3) centric point
x2, y2 = make_gaussian_quantiles(mean=(3,3), n_samples=500, n_features=2, n_classes=2)
x_data = np.concatenate((x1, x2))
y_data = np.concatenate((y1, -y2 + 1))

dtree = tree.DecisionTreeClassifier(max_depth=3)
dtree.fit(x_data, y_data)
print("Decision tree's accuracy is: ", dtree.score(x_data, y_data))

# Adaboost
# 1. Initial weight is 1/m, m is number of dataset
# 2. For each epochs(k)
#  Randomly extract subset with replacement, train model and calculate error rate
#  Correct classifications will reduce weight, while Wrong classifications are by contract
#                          //weight * error(Mi)/(1 - error(Mi))
# 3. Increase weight of Correct models, while Wrong models are by contract. Voting from k classifiers.
#                          // wi = log((1 - error(Mi))/error(Mi))
ada_tree = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=50)
ada_tree.fit(x_data, y_data)
print("Adaboost tree's accuracy is: ", ada_tree.score(x_data, y_data))

def plot(model):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z)


plt.subplot(131)
plt.title('Original Data')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.subplot(132)
plot(dtree)
plt.title('Decision Tree')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.subplot(133)
plot(ada_tree)
plt.title('Adaboost Tree')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.show()