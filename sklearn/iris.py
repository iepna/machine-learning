# iris数据加载
from sklearn import datasets
iris = datasets.load_iris()

# 显示iris数据
print(iris.data)
print(iris.feature_names)
print(iris.target)

# 确认数据类型
print(type(iris.data))
print(type(iris.target))
print(iris.data.shape)
print(iris.target.shape)

# 数据赋值
x = iris.data
y = iris.target

# knn模型
# 模型调用
from sklearn.neighbors import KNeighborsClassifier

# 创建实例
knn = KNeighborsClassifier(n_neighbors=1)
# 训练
knn.fit(x, y)
# 预测
print(knn.predict([[1, 2, 3, 4], [2, 4, 1, 3]]))

# 设定一个新得k值进行knn建模
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(x, y)
print(knn_5.predict([[1, 2, 3, 4], [2, 4, 1, 2]]))
y_pred_5 = knn_5.predict(x)
y_pred_1 = knn.predict(x)

# 模型的评估
# 准确率
from sklearn.metrics import accuracy_score
print(accuracy_score(y, y_pred_1))
print(accuracy_score(y, y_pred_5))

# 分离训练数据与测试数据
# 数据分离
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# 分离后数据集的维度确认
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 分离后数据集的训练与评估
knn_5_s = KNeighborsClassifier(n_neighbors=5)
knn_5_s.fit(x_train, y_train)
y_train_pred = knn_5_s.predict(x_train)
y_test_pred = knn_5_s.predict(x_test)

# 分离后模型预测的准确率
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))

# 确定更合适的k值
# k：1-25
# 遍历所有可能的参数组合
# 建立响应的model
# model训练 》 预测
# 给与测试数据的准确率的计算
# 查看最高的准确率对应的k
k_range = list(range(1, 26))
score_train = []
score_test = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_train_pred = knn.predict(x_train)
    y_test_pred = knn.predict(x_test)
    score_train.append(accuracy_score(y_train, y_train_pred))
    score_test.append(accuracy_score(y_test, y_test_pred))
for k in k_range:
    print(k, score_train[k-1])

for k in k_range:
    print(k, score_test[k-1])
