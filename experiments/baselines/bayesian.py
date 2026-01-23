import sys
import os
import optuna
import logging

# 适配路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from core_refactor.base import Optimizer

# 关闭 Optuna 的部分啰嗦日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BayesianOptimizer(Optimizer):
    def __init__(self, evaluator, param_space, max_evals=100):
        super().__init__(evaluator, param_space)
        self.max_evals = max_evals

    def optimize(self, video_sequences):
        print(f"=== Bayesian Optimization (TPE) Started (Budget: {self.max_evals}) ===")

        def objective(trial):
            # 1. 将 SearchSpace 映射到 Optuna Trial
            # 遍历所有模块，调用 params.py 中定义的 to_optuna 接口
            for m_name, module in self.param_space.modules.items():
                module.to_optuna(trial)

            # 2. 从 Trial 获取当前采样的扁平参数字典
            # e.g., {'vaq/aq-mode': 1, 'cutree/cutree-strength': 0.5 ...}
            flat_params = trial.params

            # 3. 更新 SearchSpace 状态
            self.param_space.update_from_flat_dict(flat_params)

            # 4. 获取结构化配置并评估
            current_config = self.param_space.get_all_config()

            # 更新日志上下文 (Trial Number)
            self.evaluator.set_context(module="BayesOpt", iteration=trial.number)

            cost = self.evaluator.evaluate(current_config, video_sequences)
            return cost

        # 创建 Optuna Study (最小化 Cost)
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        # 显式加入默认参数作为起始点 (Warm Start)
        # 这是一个良好的实践，告诉 BO "先看看默认参数效果如何"
        default_params = {}
        for m_name, module in self.param_space.modules.items():
            for p_name, p in module.params.items():
                default_params[f"{m_name}/{p_name}"] = p.value
        study.enqueue_trial(default_params)

        # 开始优化
        study.optimize(objective, n_trials=self.max_evals)

        print(f"=== Bayesian Opt Finished. Best Cost: {study.best_value:.4f} ===")
        print(f"Best Params: {study.best_params}")

        # 将 ParamSpace 恢复到最优状态并返回
        self.param_space.update_from_flat_dict(study.best_params)
        return self.param_space.get_all_config()
