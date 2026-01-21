import copy
from .base import Optimizer


class RelevancyGuidedOptimizer(Optimizer):
    def __init__(self, evaluator, param_space, module_order=None):
        super().__init__(evaluator, param_space)
        # 默认排序 (Task 1 Result): Cutree -> VAQ -> Qcomp -> Psyrdo -> Psyrdoq
        # 你可以通过传入 module_order 来进行消融实验（如传入逆序或随机序）
        self.module_order = (
            module_order
            if module_order
            else ["psyrdo", "psyrdoq", "qcomp", "vaq", "cutree"]
        )
        self.epsilon = 0.001  # 收敛阈值

    def optimize(self, video_sequences):
        """
        Task 2: 离线最优参数搜索
        针对给定的 video_sequences (代表一个特定的 target bitrate/quality)
        """
        best_global_config = self.param_space.get_all_config()
        min_global_cost = self.evaluator.evaluate(best_global_config, video_sequences)

        iteration = 0
        while True:
            iteration += 1
            cost_at_start = min_global_cost
            print(
                f"\n=== Global Iteration {iteration} (Start Cost: {cost_at_start:.4f}) ==="
            )

            # 按照相关性顺序遍历模块
            for module_name in self.module_order:
                print(f"--- Optimizing Module: {module_name} ---")
                module = self.param_space.modules[module_name]

                # 获取当前配置用于搜索
                current_cost = min_global_cost

                if module.is_dual:
                    # 双参数模块：交替优化 (Coordinate Descent within Module)
                    new_cost = self._optimize_dual_param_module(
                        module, video_sequences, current_cost
                    )
                else:
                    # 单参数模块：直接方向搜索
                    # 注意：对于CUTree，虽然有cutree开关和strength，但强度0等效于关闭
                    # 这里假设只优化 strength (index 1)，开关 (index 0) 保持开启
                    # 如果需要同时优化开关，可以视为Dual，或者在strength list里包含0
                    target_param = list(module.params.values())[
                        0
                    ]  # 默认取第一个主要参数
                    if len(module.params) > 1:
                        # 针对 cutree 特殊处理，取 strength 参数
                        target_param = module.params.get(
                            "cutree-strength", list(module.params.values())[0]
                        )

                    new_cost = self._directional_search(
                        module.name, target_param, video_sequences, current_cost
                    )

                min_global_cost = new_cost

            # 全局收敛检测
            cost_drop = cost_at_start - min_global_cost
            print(
                f"=== Global Iteration {iteration} End. Cost Drop: {cost_drop:.6f} ==="
            )

            if cost_drop < self.epsilon:
                print("Global convergence reached.")
                break

        return self.param_space.get_all_config()

    def _optimize_dual_param_module(self, module, videos, current_cost):
        """
        针对双参数模块 (e.g., VAQ: mode + strength) 的内部循环
        策略：先优化 Strength (连续)，再优化 Mode (离散)，循环直到收敛
        """
        # 识别参数类型
        # 假设 params 字典里，一个是 'mode' 类 (离散少)，一个是 'strength' 类 (离散多/连续)
        # 这里通过名字简单判断，或者在 Module 定义时指定
        param_list = list(module.params.values())
        # 简单的启发式：candidates 数量少的作为 mode，多的作为 strength
        p1, p2 = param_list[0], param_list[1]
        if len(p1.candidates) < len(p2.candidates):
            mode_param, str_param = p1, p2
        else:
            mode_param, str_param = p2, p1

        local_cost = current_cost

        while True:
            prev_cost = local_cost

            # 1. Optimize Strength (Directional Search)
            # print(f"  [Inner] Optimizing Strength: {str_param.name}")
            local_cost = self._directional_search(
                module.name, str_param, videos, local_cost
            )

            # 2. Optimize Mode (Exhaustive Search / Traversal)
            # print(f"  [Inner] Optimizing Mode: {mode_param.name}")
            local_cost = self._traversal_search(
                module.name, mode_param, videos, local_cost
            )

            if abs(prev_cost - local_cost) < 1e-6:
                break

        return local_cost

    def _directional_search(self, module_name, param, videos, current_cost):
        """
        单参数搜索：按照性能上升方向搜索 (Hill Climbing)
        替代原有的 三分+退火
        """
        # 1. 尝试向左走 (Value 变小)
        neighbors = param.get_neighbors()

        direction = 0  # 0: stay, -1: left, 1: right

        # Test Left
        if "left" in neighbors:
            # 临时修改参数值进行评估
            original_idx = param.idx
            param.move_index(-1)
            cost_left = self.evaluator.evaluate(
                self.param_space.get_all_config(), videos
            )

            if cost_left < current_cost:
                direction = -1
                current_cost = cost_left
            else:
                # Revert
                param.idx = original_idx

        # 如果左边没变好，尝试右边
        if direction == 0 and "right" in neighbors:
            original_idx = param.idx
            param.move_index(1)
            cost_right = self.evaluator.evaluate(
                self.param_space.get_all_config(), videos
            )

            if cost_right < current_cost:
                direction = 1
                current_cost = cost_right
            else:
                param.idx = original_idx

        # 如果确定了方向，就沿着该方向一直走，直到 Cost 不再下降
        if direction != 0:
            while True:
                original_idx = param.idx
                # 尝试再走一步
                new_val = param.move_index(direction)

                # 如果已经到边界，move_index 不会改变 idx
                if param.idx == original_idx:
                    break

                new_cost = self.evaluator.evaluate(
                    self.param_space.get_all_config(), videos
                )
                if new_cost < current_cost:
                    current_cost = new_cost
                    # print(f"    -> Improved {param.name} to {new_val}, Cost: {current_cost:.4f}")
                else:
                    # 性能变差了，回退一步，并结束
                    param.idx = original_idx
                    break

        return current_cost

    def _traversal_search(self, module_name, param, videos, current_cost):
        """
        模式遍历搜索：因为 Mode 选项很少，直接遍历比较快
        """
        best_idx = param.idx
        min_cost = current_cost

        original_idx = param.idx

        for i in range(len(param.candidates)):
            if i == original_idx:
                continue

            param.idx = i
            cost = self.evaluator.evaluate(self.param_space.get_all_config(), videos)

            if cost < min_cost:
                min_cost = cost
                best_idx = i

        param.idx = best_idx
        return min_cost
