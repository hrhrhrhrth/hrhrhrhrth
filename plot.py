import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False

def function_draw():
    # 1. 读取数据
    data = np.loadtxt("WIFI.txt")
    X = data[:, :-1]
    y = data[:, -1]

    # 2. 定义特征和类别的数量
    num_features = X.shape[1]
    num_classes = 4

    # 3. 划分特征的区间
    intervals = [-100, -80, -60, -40, -20, 0]

    # 4. 创建一个3x3的多子图布局
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # 5. 遍历每个特征并生成柱状图
    for feature_idx in range(num_features):
        row, col = divmod(feature_idx, 3)
        ax = axs[row, col]

        feature_data = X[:, feature_idx]

        # 统计每个区间内分到各个类别的样本数量
        class_counts = np.zeros((len(intervals) - 1, num_classes), dtype=int)

        for i in range(len(intervals) - 1):
            lower_bound = intervals[i]
            upper_bound = intervals[i + 1]
            mask = (feature_data >= lower_bound) & (feature_data < upper_bound)
            subset_y = y[mask]
            for j in range(1, num_classes + 1):
                class_counts[i, j - 1] = np.sum(subset_y == j)

        # 绘制柱状图
        width = 0.2
        for i in range(num_classes):
            x = np.arange(len(intervals) - 1) + i * width
            bars = ax.bar(x, class_counts[:, i], width, label=f'类别 {i + 1}')

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_xlabel(f"划分区间")
        ax.set_ylabel("样本数量")
        ax.set_title(f"特征{feature_idx + 1}区间划分分类情况")
        ax.set_xticks(np.arange(len(intervals) - 1) + width)
        ax.set_xticklabels([f"{intervals[i]} to {intervals[i + 1]}" for i in range(len(intervals) - 1)])
        ax.legend()

        # 设置Y轴最大坐标为600
        ax.set_ylim(0, 600)

    # 删除多余的子图
    for feature_idx in range(num_features, 9):
        row, col = divmod(feature_idx, 3)
        fig.delaxes(axs[row, col])

    # 调整布局，以防止重叠
    plt.tight_layout()

    # 显示图形
    plt.show()