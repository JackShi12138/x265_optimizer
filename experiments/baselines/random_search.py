import sys
import os
import logging

# 适配路径，确保能导入 core_refactor
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from core_refactor.base import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, evaluator, param_space, max_evals=100):
        super().__init__(evaluator, param_space)
        self.max_evals = max_evals  # 计算预算：最大评估次数

    def optimize(self, video_sequences):
        """
        随机搜索主循环
        """
        print(f"=== Random Search Started (Budget: {self.max_evals}) ===")

        best_config = self.param_space.get_all_config()
        # 初始 Cost (默认参数)
        self.evaluator.set_context(module="Random", iteration=0)
        min_cost = self.evaluator.evaluate(best_config, video_sequences)
        print(f"[Random] Initial Cost: {min_cost:.4f}")

        for i in range(1, self.max_evals + 1):
            # 1. 随机采样整个参数空间
            # 注意：这会直接修改 param_space 内部的 current_idx
            current_config = self.param_space.random_sample()

            # 2. 评估
            self.evaluator.set_context(module="Random", iteration=i)
            cost = self.evaluator.evaluate(current_config, video_sequences)

            # 3. 更新最优解
            if cost < min_cost:
                print(
                    f"    [Iter {i}] New Best Found! Cost: {cost:.4f} (was {min_cost:.4f})"
                )
                min_cost = cost
                best_config = current_config  # 这里其实需要 deepcopy，但因为 evaluate 没改对象，暂且这样
            else:
                # print(f"    [Iter {i}] Cost: {cost:.4f}") # 可选：打印每次结果
                pass

        print(f"=== Random Search Finished. Best Cost: {min_cost:.4f} ===")
        return best_config
