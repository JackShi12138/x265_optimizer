import sys
import os
import unittest

# 路径适配
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.params import SearchSpace


class TestPhase1Features(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace()

    def test_random_sample(self):
        """验证随机采样功能"""
        print("\n=== Testing Random Sample ===")
        # 1. 获取默认配置
        self.space._init_default_space()
        default_aq_mode = self.space.modules["vaq"].params["aq-mode"].value

        # 2. 随机采样多次，确保值发生了变化 (概率极高)
        changed = False
        for _ in range(10):
            self.space.random_sample()
            new_val = self.space.modules["vaq"].params["aq-mode"].value
            if new_val != default_aq_mode:
                changed = True
                break

        if changed:
            print(
                f"[Pass] Random sample successfully changed parameter value: {new_val}"
            )
        else:
            print(
                "[Warning] Random sample did not change value in 10 tries (unlikely but possible)"
            )

        # 检查是否还在合法范围内
        val = self.space.modules["vaq"].params["aq-mode"].value
        self.assertIn(val, [0, 1, 2, 3, 4])

    def test_flat_dict_update(self):
        """验证从扁平字典还原状态的功能 (模拟 Optuna 返回值)"""
        print("\n=== Testing Flat Dict Update ===")

        # 模拟一个 Optuna 返回的最优解字典
        optuna_best_params = {
            "vaq/aq-mode": 4,
            "cutree/cutree-strength": 0.5,
            "psyrdo/psy-rd": 4.0,
        }

        # 执行更新
        self.space.update_from_flat_dict(optuna_best_params)

        # 验证内部状态是否改变
        curr_config = self.space.get_all_config()

        self.assertEqual(curr_config["vaq"]["aq-mode"], 4)
        self.assertAlmostEqual(curr_config["cutree"]["cutree-strength"], 0.5)
        self.assertAlmostEqual(curr_config["psyrdo"]["psy-rd"], 4.0)

        print(f"[Pass] Successfully updated state from flat dict: {optuna_best_params}")


if __name__ == "__main__":
    unittest.main()
