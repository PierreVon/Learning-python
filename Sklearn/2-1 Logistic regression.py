import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import linear_model

x_data = np.linspace(-1, 1, 20)
noize = 0.3
y_data = np.random.rand(20) + ([noize]*10 + [-noize]*10)
data = np.hstack((x_data.reshape(-1,1),y_data.reshape(-1,1)))
print(data.shape)
label_data = [0]*10 + [1]*10

logistic = linear_model.LogisticRegression()
logistic.fit(data, label_data)

s0 = plt.scatter(x_data[:9],y_data[:9] ,c='b')
s1 = plt.scatter(x_data[10:],y_data[10:] ,c='r')

x_test = np.array([[-1],[1]])
# logistic.intercept_ + x1 * logistic.coef_[0][0] + x2 * logistic.coef_[0][1] = 0
# W0 + X1 * W1 + X2 * W2 = 0, decision boundary
y_test = (-logistic.intercept_ - x_test * logistic.coef_[0][0] )/logistic.coef_[0][1]
plt.plot(x_test, y_test, 'k')

plt.legend(handles = [s0,s1],labels=['label0','label1'], loc='best')
plt.show()

prediction = logistic.predict(data)
print(classification_report(label_data, prediction))