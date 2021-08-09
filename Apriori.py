from copy import deepcopy


class Apriori:
    def __init__(self, support, conf):
        """
        Parameters:
            support: 支持度
            conf: 置信度
        """
        self.support = support
        self.conf = conf

    def get_frist_set(self, data):
        """ 得到所有种类 """
        s = set()
        for row in data:
            s = s | row
        return [[k] for k in s]

    def isin(self, items, row):
        """ 
        判断项集是否在数据集中

        Parameters:
            items: 候选集
            row: 数据集的一行
        """
        for x in items:
            if x not in row:
                return False
        return True

    def find_frequent(self, data, compose_data):
        """
        发现频繁项目集

        Parameters:
            data: 数据集
            compose_data: 频繁项集候选集
        Returns:
            res: 频繁项集
        """
        res = {}
        for t in compose_data:
            res[",".join(t)] = 0

        # 遍历数据库计算所有项集出现的次数
        for row in data:
            for items in compose_data:
                if self.isin(items,row):
                    res[",".join(items)] += 1

        # 删除小于支持度的项集
        for k in list(res.keys()):
            if res[k] < self.support_length:
                del res[k]
        return res

    def merge(self, a, b):
        """ 合并两个频繁项集 """
        return list(set(a + b))

    def compose(self, pre_frequent ,level):
        """
        交叉生成下一阶段备选的频繁项集

        Parameters:
            pre_frequent: 上一阶段的频繁项集
            level: 频繁项集其中的数据数量
        Returns:
            res: 频繁项候选集
        """
        frequent_items = [k.split(",") for k in pre_frequent.keys()]
        compose_data = list()
        for i in range(len(frequent_items)):
            for j in range(i+1,len(frequent_items)):
                next_frequent = self.merge(frequent_items[i],frequent_items[j])
                if len(next_frequent) == level:
                    next_frequent.sort()
                    compose_data.append(next_frequent)

        res = {}
        re_compose_data = []
        for x in compose_data:
            res[",".join(x)] = True
        compose_data = [k.split(",") for k in res.keys()]
        return list(compose_data)

    def search_count(self, search_itme, all_frequent):
        for items in all_frequent:
            res = items.get(search_itme)
            if res != None:
                return res
        return None


    def fit_predict(self, data):
        """ 核心函数，调用执行Apriori算法 """
        self.support_length = self.support * len(data)
        all_frequent = list()

        compose_data = self.get_frist_set(data)
        i = 2

        print("频繁项集有:")
        while compose_data:
            frequent = self.find_frequent(data,compose_data)
            print(f"第{i-1}频繁项集为:",frequent)
            compose_data = self.compose(frequent,i)
            all_frequent.append(frequent)
            i += 1

        print("\n强关联规则有:")
        for index, frequent in enumerate(all_frequent):
            if index == 0:
                continue
            for items in frequent:
                all_item_count = self.search_count(items,all_frequent)
                items = items.split(",")
                for del_item in items:
                    copy_items = deepcopy(items)
                    copy_items.remove(del_item)
                    copy_items_str = ",".join(copy_items)
                    del_item_count = self.search_count(copy_items_str,all_frequent)
                    calc_conf = all_item_count / del_item_count
                    if calc_conf > self.conf:
                        print(copy_items_str, "==> ", del_item, "置信度为:",calc_conf)


if __name__ == "__main__":
    data = [
        {"棒棒糖","啤酒","雪碧"},
        {"尿布","啤酒","可乐"},
        {"棒棒糖","尿布","啤酒","可乐"},
        {"尿布","可乐"},
    ]

    apriori = Apriori(support=0.5, conf = 0.8)
    apriori.fit_predict(data)
