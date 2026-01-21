import sys
import os
import unittest
from copy import deepcopy

# 将项目根目录加入路径，确保能导入 core_refactor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.base import CostEvaluator
from core_refactor.params import SearchSpace
from core_refactor.algorithms import RelevancyGuidedOptimizer


class MockEvaluator(CostEvaluator):
    """
    一个伪造的评估器，它不运行 x265。
    它的“成本”是当前参数与【预设目标参数】之间的距离平方和。
    优化器应该能找到让 Cost 为 0 的那组参数。
    """

    def __init__(self, target_config):
        self.target_config = target_config
        self.call_count = 0
        self.history = []

    def evaluate(self, params, video_sequences=None):
        self.call_count += 1
        total_dist = 0.0

        # 计算所有参数与目标的欧氏距离平方
        for m_name, m_params in params.items():
            for p_name, p_val in m_params.items():
                target_val = self.target_config[m_name][p_name]
                # 简单平方误差：模拟凸函数
                total_dist += (p_val - target_val) ** 2

        self.history.append(total_dist)
        return total_dist

    def reset(self):
        self.call_count = 0
        self.history = []


class TestOptimizerLogic(unittest.TestCase):
    def setUp(self):
        # 1. 初始化标准搜索空间
        self.space = SearchSpace()

        # 2. 设定一个“上帝视角”的最优解 (Target)
        # 我们故意选一个不在默认位置的值，测试算法能不能“爬”过去
        self.target_config = self.space.get_all_config()

        # 修改目标值 (确保这些值都在 SearchSpace 的 candidates 列表中)
        # VAQ (Dual): 改 Strength=1.5 (默认1.0), Mode=1 (默认2)
        self.target_config["vaq"]["aq-strength"] = 1.5
        self.target_config["vaq"]["aq-mode"] = 1

        # CUTree (Single): 改 Strength=1.0 (默认2.0)
        self.target_config["cutree"]["cutree-strength"] = 1.0

        # QComp (Single): 改 0.8 (默认0.6)
        self.target_config["qcomp"]["qcomp"] = 0.8

        # 3. 初始化 Mock 评估器
        self.evaluator = MockEvaluator(self.target_config)

        # 4. 初始化优化器
        # 按照修正后的顺序: psyrdo -> psyrdoq -> qcomp -> vaq -> cutree
        self.module_order = ["psyrdo", "psyrdoq", "qcomp", "vaq", "cutree"]
        self.optimizer = RelevancyGuidedOptimizer(
            self.evaluator, self.space, module_order=self.module_order
        )

    def test_convergence_to_target(self):
        """测试优化器能否从默认值准确搜索到我们设定的 Target 值"""
        print("\n=== Running Mock Optimization Test ===")

        # 只需要传一个空字典，因为 MockEvaluator 不看 video_sequences
        # 注意：每次优化前重置空间到默认值，防止污染
        self.space._init_default_space()

        best_config = self.optimizer.optimize(video_sequences={"mock": 0})

        print(f"\nOptimization Finished in {self.evaluator.call_count} steps.")

        # --- 验证逻辑 ---
        # 1. 验证 VAQ (双参数模块) 是否找到了 Strength=1.5, Mode=1
        self.assertAlmostEqual(best_config["vaq"]["aq-strength"], 1.5, places=2)
        self.assertEqual(best_config["vaq"]["aq-mode"], 1)
        print("[Pass] VAQ (Dual Parameter) converged correctly.")

        # 2. 验证 CUTree (单参数)
        self.assertAlmostEqual(best_config["cutree"]["cutree-strength"], 1.0, places=2)
        print("[Pass] CUTree (Single Parameter) converged correctly.")

        # 3. 验证 QComp
        self.assertAlmostEqual(best_config["qcomp"]["qcomp"], 0.8, places=2)
        print("[Pass] QComp converged correctly.")

        # 4. 验证最终 Cost 是否接近 0
        final_cost = self.evaluator.evaluate(best_config)
        self.assertLess(final_cost, 0.0001, "Final cost should be near zero")
        print(f"[Pass] Final Cost is {final_cost} (Optimal).")


if __name__ == "__main__":
    unittest.main()
