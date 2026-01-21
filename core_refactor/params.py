import numpy as np
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
        if value in self.candidates:
            self.idx = self.candidates.index(value)
        else:
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


class Module:
    def __init__(self, name, params: list, is_dual=False):
        self.name = name
        self.params = {p.name: p for p in params}
        self.is_dual = is_dual  # 区分单参数和双参数模块

    def get_config(self):
        return {name: p.value for name, p in self.params.items()}


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
                Parameter("aq-mode", [0, 1, 2, 3, 4], 2),  # Default idx 2 -> mode 2
                Parameter("aq-strength", drange(0.0, 3.0, 0.1), 10),  # Default 1.0
            ],
            is_dual=True,
        )

        # 2. CUTree (Single equivalent, though technically has mode)
        # Note: In your logic, cutree=0 is off, cutree=1 is on.
        # Strength search implies cutree=1. If strength=0, equivalent to off.
        self.modules["cutree"] = Module(
            "cutree",
            [
                Parameter("cutree", [0, 1], 1),
                Parameter("cutree-strength", drange(0.0, 2.5, 0.1), 20),  # Default 2.0
            ],
            is_dual=False,
        )

        # 3. Psy-RDO (Dual)
        self.modules["psyrdo"] = Module(
            "psyrdo",
            [
                Parameter("rd", [1, 2, 3, 5], 2),  # Default 3 (index 2)
                Parameter("psy-rd", drange(0.0, 5.0, 0.1), 20),  # Default 2.0
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
                ),  # Default 1.0
            ],
            is_dual=True,
        )

        # 5. QComp (Single)
        self.modules["qcomp"] = Module(
            "qcomp",
            [Parameter("qcomp", drange(0.5, 1.0, 0.01), 10)],  # Default 0.6
            is_dual=False,
        )

    def get_all_config(self):
        """获取当前所有模块的完整参数配置"""
        config = {}
        for m_name, module in self.modules.items():
            config[m_name] = module.get_config()
        return config

    def update_module_param(self, module_name, param_name, value):
        self.modules[module_name].params[param_name].set_value(value)
