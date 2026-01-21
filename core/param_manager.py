class ParameterManager:
    def __init__(self, param_ranges, initial_values):
        """
        :param param_ranges: 参数字典
        :param initial_values: 初始参数值
        """
        self.params = {}
        for module, params in param_ranges.items():
            self.params[module] = {}
            for param in params:
                assert (
                    initial_values[module][param] in params[param]
                ), f"初始值 {param}={initial_values[param]} 不在允许范围内"
                self.params[module][param] = {
                    "values": params[param],
                    "current": initial_values[module][param],
                }

    def get_current_values(self):
        result = {}
        # 遍历外层字典，键为 module
        for module, params in self.params.items():
            result[module] = {}
            # 遍历内层字典，键为参数名
            for param, param_info in params.items():
                # 获取当前参数的当前值
                result[module][param] = param_info["current"]
        return result

    def set_param_value(self, module, param, value):
        assert module in self.params, f"模块 {module} 不存在"
        assert param in self.params[module], f"参数 {param} 在模块 {module} 中不存在"
        assert (
            value in self.params[module][param]["values"]
        ), f"值 {value} 不在参数 {param}（模块 {module}）的允许范围内"
        self.params[module][param]["current"] = value
