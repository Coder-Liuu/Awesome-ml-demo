from matplotlib import pyplot as plt
from itertools import cycle, islice
import numpy as np

# 对聚类结果进行可视化
def plot_cluster(X, y):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', ]), int(max(y) + 1))))
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y])
    plt.title("Spectral Clustering")
    plt.show()
