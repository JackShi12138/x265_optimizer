import sys
import os
import random
import copy
import time

# 适配路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from core_refactor.base import Optimizer


class GeneticOptimizer(Optimizer):
    def __init__(
        self,
        evaluator,
        param_space,
        pop_size=20,
        generations=10,
        cx_prob=0.7,
        mut_prob=0.2,
    ):
        """
        :param pop_size: 种群大小 (每代有多少个个体)
        :param generations: 迭代代数
        :param cx_prob: 交叉概率
        :param mut_prob: 变异概率 (每个个体发生变异的可能性)
        """
        super().__init__(evaluator, param_space)
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

    def optimize(self, video_sequences):
        """
        遗传算法主循环
        """
        print(
            f"=== Genetic Algorithm Started (Pop: {self.pop_size}, Gens: {self.generations}) ==="
        )

        # 1. 初始化种群
        population = self._init_population()
        best_individual = None
        min_cost = float("inf")

        # 初始评估
        print(f"[GA] Evaluating Initial Population...")
        fitnesses = []
        for i, ind in enumerate(population):
            self.evaluator.set_context(module="GA_Init", iteration=0)
            cost = self.evaluator.evaluate(ind, video_sequences)
            fitnesses.append(cost)

            if cost < min_cost:
                min_cost = cost
                best_individual = copy.deepcopy(ind)

        print(f"[GA] Generation 0 Best Cost: {min_cost:.4f}")

        # 2. 进化循环
        for gen in range(1, self.generations + 1):
            # --- 选择 (Selection) ---
            # 保留精英 (Elitism): 直接把这一代最好的保留到下一代，防止退化
            next_generation = [best_individual]

            # 锦标赛选择填满剩余位置
            while len(next_generation) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                # --- 交叉 (Crossover) ---
                offspring = self._crossover(p1, p2)

                # --- 变异 (Mutation) ---
                offspring = self._mutate(offspring)

                next_generation.append(offspring)

            population = next_generation

            # --- 评估新一代 (Evaluation) ---
            fitnesses = []
            current_gen_best_cost = float("inf")

            for i, ind in enumerate(population):
                # 更新日志上下文
                self.evaluator.set_context(module="GA_Evol", iteration=gen)

                cost = self.evaluator.evaluate(ind, video_sequences)
                fitnesses.append(cost)

                if cost < current_gen_best_cost:
                    current_gen_best_cost = cost

                # 更新全局最优
                if cost < min_cost:
                    min_cost = cost
                    best_individual = copy.deepcopy(ind)
                    print(f"    [Gen {gen}] New Global Best! Cost: {min_cost:.4f}")

            print(
                f"[GA] Generation {gen} Finished. Best in Gen: {current_gen_best_cost:.4f} | Global Best: {min_cost:.4f}"
            )

        print(f"=== Genetic Algorithm Finished. Best Cost: {min_cost:.4f} ===")
        return best_individual

    def _init_population(self):
        """生成初始随机种群"""
        pop = []
        # 总是包含一个默认参数个体 (Warm Start)
        self.param_space.reset()
        pop.append(self.param_space.get_all_config())

        for _ in range(self.pop_size - 1):
            # random_sample 会直接修改 param_space 内部状态
            # get_all_config 会返回一个新的字典副本，所以是安全的
            config = self.param_space.random_sample()
            pop.append(config)
        return pop

    def _tournament_select(self, population, fitnesses, k=3):
        """锦标赛选择：随机选 k 个，取其中适应度最好(Cost最小)的"""
        candidates_indices = random.sample(range(len(population)), k)
        best_idx = -1
        best_fitness = float("inf")

        for idx in candidates_indices:
            if fitnesses[idx] < best_fitness:
                best_fitness = fitnesses[idx]
                best_idx = idx

        return population[best_idx]

    def _crossover(self, p1, p2):
        """
        均匀交叉 (Uniform Crossover)
        以 Module 为粒度进行交换。
        """
        if random.random() > self.cx_prob:
            # 不发生交叉，随机返回一个父代副本
            return copy.deepcopy(p1)

        child = {}
        # 遍历所有模块
        for module_name in p1.keys():
            # 50% 概率来自父亲，50% 概率来自母亲
            if random.random() < 0.5:
                child[module_name] = copy.deepcopy(p1[module_name])
            else:
                child[module_name] = copy.deepcopy(p2[module_name])
        return child

    def _mutate(self, individual):
        """
        单点变异 (Single Point Mutation)
        随机选择一个模块的一个参数进行重置。
        """
        if random.random() > self.mut_prob:
            return individual

        # 深拷贝以防修改原引用
        mutant = copy.deepcopy(individual)

        # 1. 随机选一个模块
        module_names = list(mutant.keys())
        target_module_name = random.choice(module_names)

        # 2. 随机选该模块下的一个参数
        # 这里我们需要利用 param_space 的能力来生成合法的新值
        # 比较麻烦的是 individual 只是字典。
        # 策略：我们将 param_space 的状态更新为 mutant 的状态，然后操作 param_space

        # 将 SearchSpace 同步到当前个体的状态
        # 注意：这里需要我们在 params.py 实现了 update_from_flat_dict 或者类似的加载逻辑
        # 简单起见，我们手动把字典值设回去
        target_module_config = mutant[target_module_name]

        # 在 SearchSpace 中找到对应的 Module 对象
        module_obj = self.param_space.modules[target_module_name]

        # 随机选一个参数对象
        param_obj_key = random.choice(list(module_obj.params.keys()))
        param_obj = module_obj.params[param_obj_key]

        # 3. 对该参数进行随机采样 (变异)
        new_val = param_obj.random_sample()

        # 4. 更新 mutant 字典
        mutant[target_module_name][param_obj_key] = new_val

        return mutant
