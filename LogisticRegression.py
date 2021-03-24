import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, tol=1e-4, max_iter=1e3, fit_intercept=True,verbose=0, seed=None):
        """
        通过梯度下降方法来求解逻辑回归的参数

        Parameters
        ----------
        lr : float
            梯度下降方法的学习率
        tol : float
            模型误差小于多少停止迭代
        max_iter : int
            最大迭代次数
        fit_intercept : bool
            是否在线性回归中，添加截距项。 默认为True
        verbose : bool
            是否打印误差的变化过程
        sedd : int or None
            随机种子
        """
        if seed:
            np.random.seed(seed)
        self.beta = None
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def fit(self, X, y):
        """
        使用梯度下降法求解模型参数
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        prev_loss = np.inf
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(self.max_iter)):
            y_prob = self.sigmoid(X @ self.beta)
            loss = self.loss(X, y, y_prob)
            if self.verbose:
                print("loss:",loss)
            if prev_loss - loss < self.tol:
                break
            prev_loss = loss
            self.beta -= self.lr * self.grad(X, y, y_prob)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        y_prob = self.sigmoid(X @ self.beta)
        return y_prob


    def predict(self, X):
        """
        使用模型进行预测
        """
        y_prob = self.predict_prob(X)
        y_pred = np.where(y_prob > 0.5,1,0)
        return y_pred

    def grad(self, X, y, y_prob):
        """
        使用梯度下降法更新参数
        """
        N, M = X.shape
        return np.dot(X.T, (y_prob - y)) / N

    
    def loss(self, X, y, y_prob):
        """
        模型的交叉熵误差
        """
        N, M = X.shape
        loss = - y * np.log(y_prob) - (1 - y) * np.log(1 - y_prob)
        return np.sum(loss) / N

    def sigmoid(self, x):
        """
        逻辑函数
        """
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("data/iris0.csv")
    data = data.sample(frac=1).reset_index(drop=True)

    labels = data.pop("class")
    y = labels.values - 1
    clf = LogisticRegression(lr=0.01)
    clf.fit(data.values,y)
    y_pred = clf.predict((data.values))
    print( sum(y == y_pred) / len(y) )

    from sklearn.linear_model import LogisticRegression as LR
    lr = LR()
    lr.fit(data.values,y)
    y_pred = lr.predict(data.values)
    print( sum(y == y_pred) / len(y) )
