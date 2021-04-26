import numpy as np

class BallTreeNode:
    def __init__(self, centroid=None, X=None, y=None):
        self.left = None
        self.right = None
        self.radius = None
        self.is_leaf = False

        self.data = X
        self.targets = y
        self.centroid = centroid

    def __repr__(self):
        fstr = "BallTreeNode(centroid={}, is_leaf={})"
        return fstr.format(self.centroid, self.is_leaf)
    
    def to_dict(self):
        d = self.__dict__
        d["id"] = "BallTreeNode"
        return d

class BallTree:
    def __init__(self, leaf_size=40, metric=None):
        """
        BallTree 数据结构

        Parameters
        ----------
        leaf_size : int
            每个叶结点的最大数据数量，默认值为40
        metric: None
            点之间的距离度量，如果没有默认使用`欧几里得度量`
        """
        self.root = None
        self.leaf_size = leaf_size
        self.metric = metric if metric is not None else np.linalg.norm

    def fit(self, X, y=None):
        """
        使用O(M log N) `K-d`构造算法递归构造一棵球树
        
        Parameters
        ----------
        X : ndarray of shape `(N, M)` 有N的样例，M个特征
        Y : ndarray of shape `(N, )` 有N个标签，或者是None,默认值为None
        """

        centroid, left_X, left_y, right_X, right_y = self._split(X, y)
        self.root = BallTree(centroid=centroid)
        self.root.radius = np.max([self.metric(centroid-x) for x in X])

        self.root.left = self._build_tree(left_X, left_y)
        self.root.right = self._build_tree(right_X, right_y)

    def _build_tree(self, X, y):
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)

        if X.shape[0] <= self.leaf_size:
            leaf = BallTreeNode(centroid,X=X,y=y)
            leaf.radius = np.max([self.metric(centroid-x) for x in X])
            leaf.is_leaf = True
            return leaf
        
        node = BallTree(centroid=centroid)
        node.radius = np.max([self.metric(centroid-x) for x in X])
        node.left = self._build_tree(left_X, left_y)
        node.right = self._build_tree(right_X, right_y)
        return node

    def _split(self, X, y=None):
        # 通过方差来寻找划分的维度
        split_dim = np.argmax(np.var(X, axis=0))

        # 在该维度上进行排序
        sort_ixs = np.argsort(X[:, split_dim])
        X, y = X[sort_ixs], y[sort_ixs] if y is not None else None

        # 计算中位数
        meadin_ix = X.shape[0] // 2
        centroids = X[meadin_ix]

        # 通过中位数进行划分数据集
        left_X, left_y = X[:meadin_ix], y[:meadin_ix] if y is not None else None
        right_X, right_y = X[meadin_ix:], y[meadin_ix:] if y is not None else None
        return centroids, left_X, left_y, right_X, right_y

# n = BallTreeNode()
# a = n.to_dict()
# print(a)
# print(n)

        
# __repr__和__str__这两个方法都是用于显示的，__str__是面向用户的，而__repr__面向程序员。
# 对象的__dict__中存储了一些self.xxx的一些东西
