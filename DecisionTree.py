import numpy as np


class Node:
    def __init__(self,left,right,rule):
        self.left = left
        self.right = right
        self.feature = rule[0]
        self.threshold = rule[1]

class Leaf:
    def __init__(self,value):
        self.value = value

class DecisionTree:
    def __init__(
        self,
        classifier=True,
        max_depth=None,
        criterion="entropy",
        seed=None,
    ):
        """
        一个可以解决分类的决策树模型

        Parameters
        ----------
        classifier : bool
            问题是分类问题(classifier = True),目前只支持分类问题,默认为True

        max_depth : int or None
            决策树最大生长的层数.如果是None,则为一棵完整的决策树

        criterion : {'entropy'} 
            决策树进行划分时,选择的依据

        seed : int or None
            生成决策树的随机种子.默认为None
        """
        if seed:
            np.random.seed(seed)

        self.depth = 0
        self.root = None

        self.classifier = classifier
        self.criterion = criterion
        self.max_depth = max_depth if max_depth else np.inf

    def fit(self, X, Y):
        """
        使用数据集,训练一个二分类树

        Parameters
        ----------
        X : class:`ndarray<numpy.ndarray>` of shape (N,M)
            训练集中一共有N个数据,每个数据集具有M个属性
        y : class:`ndarray<numpy.ndarray>` of shape (N,)
            对于训练集X中所有数据的标签
        """
        self.n_classes = max(Y) + 1
        self.n_feats = X.shape[1]
        self.root = self._grow(X,Y)

    def _grow(self,X,Y,cur_depth=0):
        """
        决策树的生长
        """
        # 成长结束时
        if len(set(Y)) == 1:
            prob = np.zeros(self.n_classes)
            prob[Y[0]] = 1.0
            return Leaf(prob)

        cur_depth += 1
        N,M = X.shape
        feat_idxs = np.random.choice(M,self.n_feats,replace=False)

        # 计算最好的特征和阈值
        feat, thresh = self._segment(X, Y, feat_idxs)
        # 划分数据集
        l = np.argwhere(X[:,feat] <= thresh).flatten()
        r = np.argwhere(X[:,feat] > thresh).flatten()
        # 进行成长
        left = self._grow(X[l,:],Y[l],cur_depth)
        right = self._grow(X[r,:],Y[r],cur_depth)

        return Node(left,right,(feat,thresh))

    def _segment(self,X,Y,feat_idxs):
        """
        根据`分割标准`选择得分最高的特征
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            # 选择出这一列来
            vals = X[:,i]
            # 计算所有的分割点
            levels = np.unique(vals)

            if (len(levels) > 1):
                thresholds = (levels[1:] + levels[:-1]) / 2 
            else:
                thresholds = levels

            gains = np.array([self._calc_gain(Y,t,vals) for t in thresholds])
            if(gains.max() > best_gain):
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]
        return split_idx, split_thresh

    def _calc_gain(self,Y,split_thresh,feat_values):
        """
        计算每个分割点的`分割标准`值
        """
        if self.criterion == "entropy":
            loss = self.entropy

        parent_loss = loss(Y)

        # 进行划分
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0;

        # 计算得分
        n = len(Y)
        n_l, n_r = len(left),len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r
        
        ig = parent_loss - child_loss
        return ig

    def predict(self,X):
        """
        使用训练好的决策树来对数据X进行分类

        Parameters
        ----------
        X : class:`ndarray <numpy.ndarray>` of shape (N,M)
            数据集X必须保证和训练集的大小一致

        Returns
        -------
        preds: class:`ndarray <numpy.ndarray>` of shape (N,)
        """
        return np.array([self._traverse(x,self.root) for x in X])

    def _traverse(self,X,node):
        if isinstance(node,Leaf):
            return node.value.argmax()
        if X[node.feature] <= node.threshold:
            return self._traverse(X,node.left)
        return self._traverse(X,node.right)

    def entropy(self,y):
        hist = np.bincount(y)
        ps = hist / np.sum(hist)
        return  -np.sum([p * np.log2(p) for p in ps if p > 0])


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("iris0.csv")
    data = data.sample(frac=1).reset_index(drop=True)

    labels = data.pop("class")
    clf = DecisionTree()
    clf.fit(data.values,labels.values)
    preds = clf.predict((data.values))
    print(labels.values)
    print(preds)
