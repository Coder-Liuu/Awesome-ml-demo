import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1, fit_intercept=True):
        """
        通过正规化方程训练岭回归模型

        Parameters
        ----------
        alpha: float
            L2惩罚项的系数，alpha越大beta越小，模型越平缓
        fit_intercept: bool
            是否给模型添加截距
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        通过正规方程求解模型系数

        Parameters
        ----------
        X : class:`ndarray<numpy.ndarray>` of shape (N,M)
            训练集中一共有N个数据,每个数据集具有M个属性
        y : class:`ndarray<numpy.ndarray>` of shape (N,)
            对于训练集X中所有数据的标签
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        A = self.alpha * np.eye(X.shape[1])
        self.beta = np.linalg.inv(X.T @ X + A ) @ X.T @ y

    def predict(self, X):
        """
        使用训练好的岭回归模型进行预测

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

