import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        """
        通过正太方程求解线性回归

        Parameters
        ----------
        fit_intercept : bool
            是否在线性回归中，添加截距项。 默认为True
        """
        self.bate = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        通过正太方程求解回归系数

        Parameters
        ----------
        X : class:`ndarray<numpy.ndarray>` of shape (N,M)
            训练集中一共有N个数据,每个数据集具有M个属性
        y : class:`ndarray<numpy.ndarray>` of shape (N,)
            对于训练集X中所有数据的标签
        """
        if self.fit_intercept:
            # 添加一列1
            X = np.c_[np.ones(X.shape[0]), X]

        # 正规化方程做法: https://blog.csdn.net/fuhuo_/article/details/109549029
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        使用训练好的线性回归模型进行预测

        Parameters
        ----------
        X : class:`ndarray<numpy.ndarray>` of shape (N,M)
            训练集中一共有N个数据,每个数据集具有M个属性

        Returns
        -------
        preds: class:`ndarray <numpy.ndarray>` of shape (N,)
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)

