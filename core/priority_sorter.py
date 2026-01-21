class PrioritySorter:
    def __init__(self, modules, module_priorities):
        """
        :param modules: 模块列表
        :param module_priorities: 模块优先级
        """
        self.modules = modules
        self.module_priorities = module_priorities

    def get_ordered_modules(self):
        # 按模块优先级降序排序
        # 复制模块列表，避免修改原始列表
        sorted_modules = self.modules.copy()
        # 使用 lambda 函数作为排序的键，根据模块优先级降序排序
        sorted_modules.sort(key=lambda x: -self.module_priorities.get(x, 0))
        return sorted_modules
