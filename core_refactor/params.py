import numpy as np
import random
from decimal import Decimal
from copy import deepcopy


class Parameter:
    def __init__(self, name, candidates, current_idx=0):
        self.name = name
        self.candidates = sorted(candidates)
        self.idx = current_idx

    @property
    def value(self):
        return self.candidates[self.idx]

    def set_value(self, value):
        if value in self.candidates:
            self.idx = self.candidates.index(value)
        else:
            try:
                closest_val = min(self.candidates, key=lambda x: abs(x - value))
                if abs(closest_val - value) < 1e-9:
                    self.idx = self.candidates.index(closest_val)
                    return
            except:
                pass
            raise ValueError(f"Value {value} not in candidates for {self.name}")

    def get_neighbors(self) -> dict:
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

    def random_sample(self):
        self.idx = random.randint(0, len(self.candidates) - 1)
        return self.value

    def to_optuna(self, trial, scope_name):
        """将参数注册到 Optuna Trial"""
        param_key = f"{scope_name}/{self.name}"
        return trial.suggest_categorical(param_key, self.candidates)


class Module:
    def __init__(self, name, params: list, is_dual=False, dependency=None):
        self.name = name
        self.params = {p.name: p for p in params}
        self.is_dual = is_dual
        # dependency: 一个 lambda 函数，输入为 mode 的值，返回 bool (Strength是否有效)
        self.dependency = dependency

    def get_config(self):
        return {name: p.value for name, p in self.params.items()}

    def random_sample(self):
        for p in self.params.values():
            p.random_sample()

    def is_strength_active(self, mode_value):
        """检查在当前 Mode 值下，Strength 参数是否生效"""
        if self.dependency:
            return self.dependency(mode_value)
        return True

    # === [修复] 补回丢失的 to_optuna 方法 ===
    def to_optuna(self, trial):
        for p in self.params.values():
            p.to_optuna(trial, self.name)

    # ======================================


class SearchSpace:
    def __init__(self):
        self.modules = {}
        self.reset()

    def reset(self):
        self.modules.clear()
        self._init_default_space()

    def _init_default_space(self):
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
        # 规则: aq-mode=0 时禁用 VAQ，aq-strength 无效
        self.modules["vaq"] = Module(
            "vaq",
            [
                Parameter("aq-mode", [0, 1, 2, 3, 4], 2),
                Parameter("aq-strength", drange(0.0, 3.0, 0.1), 10),
            ],
            is_dual=True,
            dependency=lambda mode: mode != 0,
        )

        # 2. CUTree (Single)
        self.modules["cutree"] = Module(
            "cutree",
            [
                Parameter("cutree", [0, 1], 1),
                Parameter("cutree-strength", drange(0.0, 2.5, 0.1), 20),
            ],
            is_dual=False,
        )

        # 3. Psy-RDO (Dual)
        # 规则: rd < 3 (即1, 2) 时，Psy-RD 无效
        self.modules["psyrdo"] = Module(
            "psyrdo",
            [
                Parameter("rd", [1, 2, 3, 5], 2),  # Default 3
                Parameter("psy-rd", drange(0.0, 5.0, 0.1), 20),
            ],
            is_dual=True,
            dependency=lambda mode: mode >= 3,
        )

        # 4. Psy-RDOQ (Dual)
        # 规则: rdoq-level=0 时，Psy-RDOQ 无效
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
            dependency=lambda mode: mode != 0,
        )

        # 5. QComp (Single)
        self.modules["qcomp"] = Module(
            "qcomp", [Parameter("qcomp", drange(0.5, 1.0, 0.01), 10)], is_dual=False
        )

    def get_all_config(self):
        config = {}
        for m_name, module in self.modules.items():
            config[m_name] = module.get_config()
        return config

    def update_module_param(self, module_name, param_name, value):
        self.modules[module_name].params[param_name].set_value(value)

    def random_sample(self):
        for m in self.modules.values():
            m.random_sample()
        return self.get_all_config()

    def update_from_flat_dict(self, flat_params):
        for key, value in flat_params.items():
            if "/" not in key:
                continue
            module_name, param_name = key.split("/", 1)
            if module_name in self.modules:
                module = self.modules[module_name]
                if param_name in module.params:
                    module.params[param_name].set_value(value)
