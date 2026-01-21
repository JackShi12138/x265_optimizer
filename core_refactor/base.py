from abc import ABC, abstractmethod
from typing import Dict, Any, List


class CostEvaluator(ABC):
    """成本评估接口，屏蔽x265运行细节"""

    @abstractmethod
    def evaluate(self, params: Dict[str, Any]) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass


class Optimizer(ABC):
    """优化器基类，适配不同的搜索策略（RG-BCD, GA, Bayes等）"""

    def __init__(self, evaluator: CostEvaluator, param_space: Any):
        self.evaluator = evaluator
        self.param_space = param_space

    @abstractmethod
    def optimize(self, video_sequences: Dict) -> Dict[str, Any]:
        pass
