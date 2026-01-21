import numpy as np
import random
from decimal import Decimal
from copy import deepcopy


class Parameter:
    def __init__(self, name, candidates, current_idx=0):
        self.name = name
        self.candidates = sorted(candidates)  # 确保有序
        self.idx = current_idx

    @property
    def value(self):
        return self.candidates[self.idx]

    def set_value(self, value):
        """通过具体值设置状态"""
        # 增加一定的容错性（针对浮点精度问题）
        if value in self.candidates:
            self.idx = self.candidates.index(value)
        else:
            # 尝试寻找最接近的候选项 (针对 Optuna 可能会返回微小误差的 float)
            try:
                closest_val = min(self.candidates, key=lambda x: abs(x - value))
                if abs(closest_val - value) < 1e-9:
                    self.idx = self.candidates.index(closest_val)
                    return
            except:
                pass
            raise ValueError(f"Value {value} not in candidates for {self.name}")

    def get_neighbors(self) -> dict:
        """获取当前值的左右邻居索引，用于方向搜索"""
        neighbors = {}
        if self.idx > 0:
            neighbors["left"] = self.candidates[self.idx - 1]
        if self.idx < len(self.candidates) - 1:
            neighbors["right"] = self.candidates[self.idx + 1]
        return neighbors

    def move_index(self, step):
        new_idx = max(0, min(len(self.candidates) - 1, self.idx + step))
        self.idx = new_idx
        return self.value

    # === Phase 1 新增接口 ===
    def random_sample(self):
        """随机选择一个值并更新当前状态"""
        self.idx = random.randint(0, len(self.candidates) - 1)
        return self.value

    def to_optuna(self, trial, scope_name):
        """
        将此参数注册到 Optuna 的 trial 中。
        使用 suggest_categorical 以确保严格选取 candidates 中的值。
        参数名格式: "scope_name/param_name"
        """
        param_key = f"{scope_name}/{self.name}"
        return trial.suggest_categorical(param_key, self.candidates)


class Module:
    def __init__(self, name, params: list, is_dual=False):
        self.name = name
        self.params = {p.name: p for p in params}
        self.is_dual = is_dual

    def get_config(self):
        return {name: p.value for name, p in self.params.items()}

    # === Phase 1 新增接口 ===
    def random_sample(self):
        for p in self.params.values():
            p.random_sample()

    def to_optuna(self, trial):
        for p in self.params.values():
            p.to_optuna(trial, self.name)


class SearchSpace:
    def __init__(self):
        self.modules = {}
        self._init_default_space()

    def _init_default_space(self):
        # 定义生成范围的辅助函数
        def drange(start, stop, step):
            d_start, d_stop, d_step = (
                Decimal(str(start)),
                Decimal(str(stop)),
                Decimal(str(step)),
            )
            r = []
            curr = d_start
            while curr <= d_stop:
                r.append(float(curr))
                curr += d_step
            return r

        # 1. VAQ (Dual)
        self.modules["vaq"] = Module(
            "vaq",
            [
                Parameter("aq-mode", [0, 1, 2, 3, 4], 2),
                Parameter("aq-strength", drange(0.0, 3.0, 0.1), 10),
            ],
            is_dual=True,
        )

        # 2. CUTree (Single equivalent)
        self.modules["cutree"] = Module(
            "cutree",
            [
                Parameter("cutree", [0, 1], 1),
                Parameter("cutree-strength", drange(0.0, 2.5, 0.1), 20),
            ],
            is_dual=False,
        )

        # 3. Psy-RDO (Dual)
        self.modules["psyrdo"] = Module(
            "psyrdo",
            [
                Parameter("rd", [1, 2, 3, 5], 2),
                Parameter("psy-rd", drange(0.0, 5.0, 0.1), 20),
            ],
            is_dual=True,
        )

        # 4. Psy-RDOQ (Dual)
        self.modules["psyrdoq"] = Module(
            "psyrdoq",
            [
                Parameter("rdoq-level", [0, 1, 2], 2),
                Parameter(
                    "psy-rdoq",
                    drange(0.0, 10.0, 0.1) + [float(i) for i in range(11, 51)],
                    10,
                ),
            ],
            is_dual=True,
        )

        # 5. QComp (Single)
        self.modules["qcomp"] = Module(
            "qcomp", [Parameter("qcomp", drange(0.5, 1.0, 0.01), 10)], is_dual=False
        )

    def get_all_config(self):
        """获取当前所有模块的完整参数配置 (嵌套字典结构)"""
        config = {}
        for m_name, module in self.modules.items():
            config[m_name] = module.get_config()
        return config

    def update_module_param(self, module_name, param_name, value):
        self.modules[module_name].params[param_name].set_value(value)

    # === Phase 1 新增接口 ===
    def random_sample(self):
        """将整个空间的所有参数随机化"""
        for m in self.modules.values():
            m.random_sample()
        return self.get_all_config()

    def update_from_flat_dict(self, flat_params):
        """
        从扁平字典更新状态 (通常用于接收 Optuna 的结果)
        flat_params: {'vaq/aq-mode': 1, 'vaq/aq-strength': 0.5, ...}
        """
        for key, value in flat_params.items():
            # 解析 "module_name/param_name"
            if "/" not in key:
                continue
            module_name, param_name = key.split("/", 1)

            if module_name in self.modules:
                module = self.modules[module_name]
                if param_name in module.params:
                    module.params[param_name].set_value(value)
