import numpy as np
import matplotlib.pyplot as plt
import plot as draw
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False

draw.function_draw()

# 1. 读取数据
data = np.loadtxt("WIFI.txt")
X = data[:, :-1]
y = data[:, -1]

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建高斯朴素贝叶斯分类器并训练模型
model = GaussianNB()
model.fit(X_train, y_train)

# 4. 使用5折交叉验证评估模型性能
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5折交叉验证
for i, score in enumerate(cv_scores):
    print(f"第{i+1}轮交叉验证准确率: {score * 100:.2f}%")
mean_accuracy = np.mean(cv_scores) * 100

print(f"5轮交叉验证平均准确率: {mean_accuracy:.2f}%")

# 5. 使用模型进行预测
y_pred = model.predict(X_test)

# 6. 评估模型性能
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"模型准确率: {accuracy:.2f}%")