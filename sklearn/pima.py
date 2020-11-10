# 皮马印第安数据集（逻辑回归）模型评估二
import pandas as pd

path = 'data/diabetes.csv'
pima = pd.read_csv(path)
print(pima.head())

# x,y赋值
feature_names = ['Pregnancies', 'Insulin', 'BMI', 'Age']
x = pima[feature_names]
y = pima['Outcome']

# 维度确认
print(x.shape)
print(y.shape)

# 数据分离
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# 模型训练
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# 测试数据集结果预测
y_pred = logreg.predict(x_test)

# 使用准确率进行评估
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# 确认正负样本数据量
print(y_test.value_counts())

# 1的比率
print(y_test.mean())

# 空准确率
print(max(y_test.mean(), 1-y_test.mean()))

# 计算并栈式混淆矩阵
print(metrics.confusion_matrix(y_test, y_pred))

# 展示部分实际结果与预测结果
print("true:", y_test.values[0:25])
print("pred:", y_pred[0:25])

# 四个以自赋值
confusion = metrics.confusion_matrix(y_test, y_pred)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

# 混淆矩阵指标
accuracy = (TP + TN)/(TP+TN+FP+FN)
mis_rate = (FP+FN)/(TP+TN+FP+FN)
recall = TP/(TP + FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)

# 综合分主
f1_score = 2*precision * recall /(precision + recall)
print(f1_score)