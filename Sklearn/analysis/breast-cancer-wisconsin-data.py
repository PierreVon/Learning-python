# compare cart with lightGBM
# compare pca with lda

import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.metrics import classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_url = "../data/breast-cancer-wisconsin-data.csv"

data = np.genfromtxt(data_url, delimiter=',', skip_header=True)
x_data = data[:500, 2:]
x_test = data[500:, 2:]
data = np.genfromtxt(data_url, dtype=str, delimiter=',', skip_header=True)
y = data[:500, 1]
y_test = data[500:, 1]

# pca = PCA(n_components=0.99)
# x_pca = pca.fit_transform(x_data)
# x_test_pca = pca.transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_data, y)
x_pca = lda.transform(x_data)
x_test_pca = lda.transform(x_test)

model = tree.DecisionTreeClassifier()
model.fit(x_pca, y)
print("--------------------Decision Tree-----------------------")
print(classification_report(y_test, model.predict(x_test_pca)))

y = list(map(lambda x: 1 if x == 'B' else 0, y))
y_test = list(map(lambda x: 1 if x == 'B' else 0, y_test))

plt.hist(x_pca)
plt.show()
# plt.scatter(x_pca[:,0], x_pca[:,1], c=y)
# plt.show()

lgb_train = lgb.Dataset(x_pca, y)  # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(x_test_pca,y_test, reference=lgb_train)  # 创建验证数据
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
evals_result = {}
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5, evals_result=evals_result)  # 训练数据需要参数列表和数据集
y_pred = gbm.predict(x_test_pca, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测

# graph = lgb.create_tree_digraph(gbm, tree_index=3, name='Tree3')
# graph.render(view=True)
# ax = lgb.plot_metric(evals_result, metric='auc')#metric的值与之前的params里面的值对应
# plt.show()
# ax = lgb.plot_importance(gbm, max_num_features=10)#max_features表示最多展示出前10个重要性特征，可以自行设置
# plt.show()
# ax = lgb.plot_tree(gbm, tree_index=3, figsize=(20, 8), show_info=['split_gain'])
# plt.show()