import numpy as np

class KMeans:
    def __init__(self, n_cluster, max_iter=300, verbose=0, seed=None):
        """
        无监督聚类算法Kmeans

        Parameters
        ----------
        n_cluster : int
            聚类簇的数量

        max_iter: int
            最大的迭代次数

        verbose: int, default=0
            是否打开过程输出

        seed : int or None
            生成决策树的随机种子.默认为None
        """
        if seed:
            np.random.seed(seed)
        self.verbose = verbose
        self.center = None
        self.max_iter = max_iter
        self.n_cluster = n_cluster

    def fit(self, X):
        """
        使用数据集,训练一个K均值聚类模型

        Parameters
        ----------
        X : class:`ndarray<numpy.ndarray>` of shape (N,M)
            训练集中一共有N个数据,每个数据集具有M个属性
        """

        n_samples, n_features = X.shape
        self.center = np.random.permutation(X)[:self.n_cluster]

        for i in range(self.max_iter):
            new_center = np.zeros((self.n_cluster,n_features))
            new_count = np.zeros(self.n_cluster)
            dis = np.array([self.distances(x) for x in X] )
            dismin = np.argmin(dis,axis=1)
            for id_,j in enumerate(dismin):
                new_center[j] += X[id_]
                new_count[j] += 1

            if self.verbose:
                print("聚类中心变化:",new_center/ (new_count[:,None]))

            if (self.center == new_center).all():
                break
            self.center = new_center

    def predict(self, X):
        """
        使用训练好的模型进行聚类

        Parameters
        ----------
        X : class:`ndarray <numpy.ndarray>` of shape (N,M)

        Returns
        -------
        preds: class:`ndarray <numpy.ndarray>` of shape (N,)
        """
        dis = np.array([self.distances(x) for x in X] )
        dismin = np.argmin(dis,axis=1)
        return dismin


    def distances(self, y):
        def Euclidean(center):
            return np.sum( (center - y)  ** 2)
        dis = np.array([ Euclidean(center) for center in self.center ])
        return dis

if __name__ == "__main__":
    data = np.array([[1,2,3],[4,5,5],[6,7,8]])
    k = KMeans(n_cluster=2,max_iter=50)
    k.fit(data)
    y = k.predict(data)
    print(y)
