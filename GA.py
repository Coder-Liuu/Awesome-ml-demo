import random
import math 
from operator import itemgetter


class GA:
    def __init__(self, paramters):
        """
        遗传算法模型  

        Parameters
        ----------
        paramters: dict
            一个传递参数的字典

        low: list 
            参数的最小值

        high: list
            参数的最大值

        popsize: int
            初始化种群的大小

        n_generations: int
            遗传算法迭代次数

        mutation_rate: float
            个体突变的概率

        crossover_rate: float
            两个个体进行交叉的概率，最大是1.0

        fitness_func: Function
            适应度函数，求解适应度的函数

        direction: {'min','max'}
            求解函数的方向，最大值或者最小值

        seed: int or None
            本次算法执行过程的随机种子
        """
        if paramters["seed"]:
            random.seed(paramters["seed"])

        self.low = paramters["low"]
        self.high = paramters["high"]
        self.popsize = paramters["popsize"]
        self.n_generations = paramters["n_generations"]
        self.mutation_rate = paramters["mutation_rate"]
        self.crossover_rate = paramters["crossover_rate"]
        self.fitness_func = paramters["fitness_func"]
        self.direction = -1 if paramters["direction"] == "min" else 1
        self.log = paramters["direction"]

    def solve(self):
        self.init_pop()

        for g in range(1, self.n_generations):
            select_pop = self.selection(self.pop, self.popsize)
            random.shuffle(select_pop)
            select_pop1 = select_pop[:self.popsize//2]
            select_pop2 = select_pop[self.popsize//2:]

            after_pop = []
            for i in range(self.popsize//2):
                p1, p2 = select_pop1[i]["Gene"], select_pop2[i]["Gene"]
                # 交叉操作
                if random.random() < self.crossover_rate:
                    p1, p2 = self.crossoperate(p1, p2)
                    # 变异操作
                    if random.random() < self.mutation_rate:
                        p1 = self.mutation(p1)
                        p2 = self.mutation(p2)
                # 更新适应度
                f1 = self.evaluate(p1, self.fitness_func)
                f2 = self.evaluate(p2, self.fitness_func)
                after_pop.append({"Gene": p1,"fitness":f1})
                after_pop.append({"Gene": p2,"fitness":f2})

            self.pop = after_pop
            self.bestindividual = self.select_best(self.pop)
            print(f"第 {g} 代: 最好的个体 {self.bestindividual}")

        print("=================================")
        x = self.select_best(self.pop)["Gene"]
        y = self.fitness_func(x)
        print("最终结果为", x, self.log, "=", y)

    def init_pop(self):
        """ 初始化种群 """
        pop = []
        for i in range(self.popsize):
            geneinfo = []
            for pos in range(len(self.low)):
                x = random.uniform(self.low[pos], self.high[pos])
                geneinfo.append(x)
            fitness = self.evaluate(geneinfo, self.fitness_func)
            pop.append({"Gene": geneinfo, "fitness": fitness})
        self.pop = pop
        self.bestindividual = self.select_best(self.pop)
        print(f"第 0 代: 最好的个体 {self.bestindividual}")

    def evaluate(self, geneinfo, func):
        """ 计算适应度 """
        return func(geneinfo) * self.direction

    def select_best(self, pop):
        """ 选择最好的一代 """
        best_geneinfo = sorted(pop, key=itemgetter("fitness"), reverse=True)[0]
        return best_geneinfo

    def selection(self, individuals, k):
        """ 个体选择(轮盘赌算法) """
        sorted_individuals = sorted(
            individuals, key=itemgetter("fitness"), reverse=True)
        sum_fits = sum(ind["fitness"] for ind in individuals)

        chosen = []
        for i in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in sorted_individuals:
                sum_ += ind["fitness"]
                if sum_ >= u:
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen

    def crossoperate(self, gene1, gene2=None):
        """ 交叉操作 """
        u = random.randint(0, len(gene1))
        tmp_gene1 = gene1[:u] + gene2[u:]
        tmp_gene2 = gene2[:u] + gene1[u:]
        return tmp_gene1, tmp_gene2

    def mutation(self, gene):
        """ 变异操作 """
        pos = random.randrange(0, len(gene))
        gene[pos] = random.uniform(self.low[pos], self.high[pos])
        return gene


# 适应度函数 or 目标函数
def fitness_func(x):
    y = math.sin(20*x[0])*(x[1]-0.3)**2
    return y

if __name__ == '__main__':
    paramters = {
        "low": [-1, -1],
        "high": [1, 1],
        "popsize": 50,
        "n_generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "fitness_func": fitness_func,
        "direction": "min",
        "seed": 9,
    }

    ga = GA(paramters)
    ga.solve()
