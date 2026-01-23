import sys
import os
import unittest
import random

# 适配路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.base import CostEvaluator
from core_refactor.params import SearchSpace
from experiments.baselines.random_search import RandomOptimizer
from experiments.baselines.genetic import GeneticOptimizer

# 尝试导入 BayesianOptimizer，如果没装 optuna 则跳过
try:
    from experiments.baselines.bayesian import BayesianOptimizer

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[Warning] Optuna not installed, skipping Bayesian tests.")


class MockCostEvaluator(CostEvaluator):
    """模拟评估器，返回一个基于参数的伪造 Cost"""

    def __init__(self):
        # 必须初始化基类以获得 log 功能支持
        # 这里我们简单覆盖 set_context 避免报错
        pass

    # === [修复] 必须实现抽象方法 reset ===
    def reset(self):
        """Mock reset: 仅作为占位符，不需要实际逻辑"""
        pass

    # ===================================

    def set_context(self, quality=None, module=None, iteration=None):
        pass  # Do nothing in mock

    def evaluate(self, params: dict, video_sequences: dict = None) -> float:
        # 简单根据参数计算一个 cost，确保同样的参数返回同样的 cost
        # 这里随便写个逻辑，比如计算所有数值参数的和
        score = 0
        for m_name, m_params in params.items():
            for p_name, val in m_params.items():
                if isinstance(val, (int, float)):
                    score += val

        # 模拟 cost：越小越好。这里假设 score 越接近 50 cost 越小
        cost = abs(score - 50) + random.random()
        return cost


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace()
        self.evaluator = MockCostEvaluator()
        # 伪造一个视频序列输入
        self.dummy_videos = {"dummy.yuv": 1000}

    def test_random_search(self):
        print("\n=== Testing Random Search ===")
        # 跑 5 次评估
        optimizer = RandomOptimizer(self.evaluator, self.space, max_evals=5)
        best_config = optimizer.optimize(self.dummy_videos)

        self.assertIsInstance(best_config, dict)
        print("[Pass] Random Search finished successfully.")

    def test_genetic_algorithm(self):
        print("\n=== Testing Genetic Algorithm ===")
        # 种群大小 4, 迭代 2 代 -> 总共约 12 次评估
        optimizer = GeneticOptimizer(
            self.evaluator,
            self.space,
            pop_size=4,
            generations=2,
            cx_prob=0.5,
            mut_prob=0.5,
        )
        best_config = optimizer.optimize(self.dummy_videos)

        self.assertIsInstance(best_config, dict)
        print("[Pass] Genetic Algorithm finished successfully.")

    def test_bayesian_optimization(self):
        print("\n=== Testing Bayesian Optimization ===")
        if not OPTUNA_AVAILABLE:
            print("[Skip] Optuna not installed.")
            return

        # 跑 5 次评估
        optimizer = BayesianOptimizer(self.evaluator, self.space, max_evals=5)
        best_config = optimizer.optimize(self.dummy_videos)

        self.assertIsInstance(best_config, dict)
        # 验证返回的配置是否合法 (检查一个 key 是否存在)
        self.assertIn("vaq", best_config)
        print("[Pass] Bayesian Optimization finished successfully.")


if __name__ == "__main__":
    unittest.main()
