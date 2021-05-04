import numpy as np
from KMeans import KMeans
from ploter.ploter import plot_cluster
np.set_printoptions(suppress=True)


class SpectralCluster:
    def __init__(self, n_clusters=2, affinity="full_link", n_neighbors=9, sigma=1.0, seed=None):
        """
        谱聚类模型

        Parameters
        ----------
        n_cluster : int
            聚类簇的数量

        affinity: {'nearest_neighbors', 'full_link'}
            建立相似矩阵的方式

        n_neighbors: int
            使用K近邻法,指定KNN算法中K的个数,
            如果affinity不使用nearest_neighbors,则不需要理会这个参数

        sigma: float, default=1.0
            高斯核的核系数

        seed : int or None
            生成模型的随机种子.默认为None
        """
        if seed:
            np.random.seed(seed)
            
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.kernel_params = 'rbf'
        self.sigma = sigma

    def fit_predict(self, X):
        """
        使用数据集,训练一个谱聚类模型，并且对数据集进行聚类

        Parameters
        ----------
        X : class:`ndarray<numpy.ndarray>` of shape (N,M)
            训练集中一共有N个数据,每个数据集具有M个属性
        """

        if self.affinity == "full_link":
            w = self.full_link(X, dist=self.rbf)
        elif self.affinity == "nearest_neighbors":
            w = self.knn_nearest(X)
        norm_laplacians = self.laplacians_matrix(w)

        eigval, eigvec = np.linalg.eig(norm_laplacians)
        ix = np.argsort(eigval)[0:self.n_clusters]
        H = eigvec[:, ix]

        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(H)
        pred = kmeans.predict(H)
        return pred

    def knn_nearest(self, x):
        k = self.n_neighbors

        l = x.shape[0]
        w = self.full_link(x, dist=np.linalg.norm)

        A = np.zeros((l, l))
        for i in range(l):
            # 对于第i个样本与其他样本的距离进行排序
            dist_with_idx = zip(w[i], range(l))
            dist_with_idx = sorted(dist_with_idx, key=lambda x: x[0])

            neighbors_id = [dist_with_idx[m][1] for m in range(k+1)]

            for j in neighbors_id:
                dist = np.exp(-w[i][j] / (2 * self.sigma * self.sigma))
                A[j][i] = A[i][j] = dist
        return A

    def rbf(self, x):
        norm = np.linalg.norm(x)
        return np.exp(-norm / (2 * self.sigma * self.sigma))

    def full_link(self, x, dist):
        l = x.shape[0]
        w = np.empty((l, l))
        for i in range(l):
            for j in range(i+1, l):
                w[j][i] = w[i][j] = dist(x[i]-x[j])
        return w

    def laplacians_matrix(self, w):
        # 计算规范化的拉普拉斯矩阵 D^(-1/2) L D^(-1/2)
        d = np.sum(w, axis=1)
        d = np.diag(d)
        laplacians = d - w
        sqrt_d = np.diag(1.0 / (np.diag(d) ** 0.5))
        norm_laplacians = sqrt_d @ laplacians @ sqrt_d
        return norm_laplacians


if __name__ == '__main__':
    x = np.load("data/circles.npy")

    y_pred = SpectralCluster(
        n_clusters=2, affinity="full_link", n_neighbors=9).fit_predict(x)
    plot_cluster(x, y_pred)
