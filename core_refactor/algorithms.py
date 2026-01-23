import copy
from .base import Optimizer


class RelevancyGuidedOptimizer(Optimizer):
    def __init__(self, evaluator, param_space, module_order=None):
        super().__init__(evaluator, param_space)
        # 默认顺序 (根据 Task 1 结论的逆序/升序)
        self.module_order = (
            module_order
            if module_order
            else ["psyrdo", "psyrdoq", "qcomp", "vaq", "cutree"]
        )
        # 移除 self.epsilon，因为不再需要判断全局收敛

    def optimize(self, video_sequences):
        """
        执行单次序列优化 (One-Pass Sequential Optimization)
        """
        best_global_config = self.param_space.get_all_config()

        # 初始评估
        self.evaluator.set_context(module="InitCheck", iteration=0)
        min_global_cost = self.evaluator.evaluate(best_global_config, video_sequences)

        print(f"=== Optimization Start (Initial Cost: {min_global_cost:.4f}) ===")

        # 核心逻辑变更：直接遍历模块列表，只执行一轮
        for i, module_name in enumerate(self.module_order):
            # 更新日志上下文
            self.evaluator.set_context(module=module_name, iteration=1)
            print(
                f"--- Optimizing Module {i+1}/{len(self.module_order)}: {module_name} ---"
            )

            module = self.param_space.modules[module_name]
            current_cost = min_global_cost

            if module.is_dual:
                # 双参数模块：内部仍保留交替最小化 (Coordinate Descent)
                # 且包含了参数依赖性检查 (is_strength_active)
                new_cost = self._optimize_dual_param_module(
                    module, video_sequences, current_cost
                )
            else:
                # 单参数模块：直接方向搜索
                target_param = list(module.params.values())[0]
                if len(module.params) > 1:
                    target_param = module.params.get(
                        "cutree-strength", list(module.params.values())[0]
                    )

                new_cost = self._directional_search(
                    module.name, target_param, video_sequences, current_cost
                )

            # 更新全局最优
            if new_cost < min_global_cost:
                diff = min_global_cost - new_cost
                min_global_cost = new_cost
                print(
                    f"    [Improved] Cost dropped by {diff:.4f} -> Current: {min_global_cost:.4f}"
                )
            else:
                print(f"    [Stable] Cost unchanged.")

        print(f"=== Optimization Finished. Final Cost: {min_global_cost:.4f} ===")

        return self.param_space.get_all_config()

    def _optimize_dual_param_module(self, module, videos, current_cost):
        """
        模块内部的双参数优化 (Strength <-> Mode)
        保留此处的 while 循环，因为模块内部参数耦合紧密，需要交替迭代收敛
        """
        param_list = list(module.params.values())
        p1, p2 = param_list[0], param_list[1]
        # 简单的启发式区分 Mode 和 Strength
        if len(p1.candidates) < len(p2.candidates):
            mode_param, str_param = p1, p2
        else:
            mode_param, str_param = p2, p1

        local_cost = current_cost

        while True:
            prev_cost = local_cost

            # 1. 优化 Strength (需检查依赖性)
            current_mode = mode_param.value
            if module.is_strength_active(current_mode):
                local_cost = self._directional_search(
                    module.name, str_param, videos, local_cost
                )

            # 2. 优化 Mode (遍历)
            local_cost = self._traversal_search(
                module.name, mode_param, videos, local_cost
            )

            # 模块内部收敛检测
            if abs(prev_cost - local_cost) < 1e-6:
                break

        return local_cost

    def _directional_search(self, module_name, param, videos, current_cost):
        """方向性搜索 (Hill Climbing)"""
        neighbors = param.get_neighbors()
        direction = 0

        # 试探左边
        if "left" in neighbors:
            original_idx = param.idx
            param.move_index(-1)
            cost_left = self.evaluator.evaluate(
                self.param_space.get_all_config(), videos
            )
            if cost_left < current_cost:
                direction = -1
                current_cost = cost_left
            else:
                param.idx = original_idx  # 回退

        # 试探右边
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
                param.idx = original_idx  # 回退

        # 锁定方向继续搜索
        if direction != 0:
            while True:
                original_idx = param.idx
                param.move_index(direction)

                # 边界检查
                if param.idx == original_idx:
                    break

                new_cost = self.evaluator.evaluate(
                    self.param_space.get_all_config(), videos
                )
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    # 性能下降，回退一步并结束
                    param.idx = original_idx
                    break

        return current_cost

    def _traversal_search(self, module_name, param, videos, current_cost):
        """遍历搜索 (针对 Mode 类参数)"""
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
