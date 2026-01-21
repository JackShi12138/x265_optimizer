import copy
import math
import random
from decimal import Decimal, getcontext
import time

getcontext().prec = 2


class ParameterOptimizer:
    def __init__(self, param_manager, cost_calculator, priority_sorter):
        self.log_file = open("search_log.txt", "w", encoding="utf-8")
        self.param_manager = param_manager
        self.cost_calculator = cost_calculator
        self.priority_sorter = priority_sorter
        self.start_time = time.time()

    def log(self, message):
        try:
            elapsed_time = time.time() - self.start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            # 只写入带有时间戳的消息
            self.log_file.write(f"[{formatted_time}] {message}\n")
            self.log_file.flush()
        except Exception as e:
            print(f"写入日志时出现错误: {e}")

    def search_vaq(self, video_sequences, param_manager):
        current_best_params = param_manager.get_current_values()
        current_best_cost = self.cost_calculator.get_optimal_loss()

        # 先测试模块关闭时的cost
        self.log("Testing vaq module off...")
        param_off = copy.deepcopy(current_best_params)
        param_off["vaq"]["aq-mode"] = 0
        cost_off = self.cost_calculator.calculate_cost(param_off, video_sequences)
        if cost_off < current_best_cost:
            current_best_cost = cost_off
            current_best_params = param_off
        # 坐标下降法搜索开启vaq模块时的最优参数
        self.log("Testing vaq module open...")
        max_iterations = 120
        strength_range = [float(Decimal(i) / Decimal(10)) for i in range(1, 31)]
        mode_range = [1, 2, 3, 4]
        param_best_open, cost_best_open = self.coordinate_descent(
            video_sequences,
            "vaq",
            param_manager,
            max_iterations,
            mode_range,
            strength_range,
        )

        if cost_best_open < current_best_cost:
            current_best_cost = cost_best_open
            current_best_params = param_best_open
        self.log(
            f"vaq module best cost: {cost_best_open}, best mode: {param_best_open['vaq']['aq-mode']}, best strength: {param_best_open['vaq']['aq-strength']}"
        )
        return current_best_params

    def search_cutree(self, video_sequences, param_manager):
        self.log("Testing cutree module...")
        strength_range = [float(Decimal(i) / Decimal(100)) for i in range(0, 26)]
        current_params = param_manager.get_current_values()
        initial_cost = self.cost_calculator.get_optimal_loss()
        initial_index = strength_range.index(
            current_params["cutree"]["cutree-strength"]
        )
        best_index, best_cost = self.simulated_annealing(
            video_sequences,
            "cutree",
            param_manager,
            strength_range,
            initial_index,
            initial_cost,
        )
        best_params = copy.deepcopy(current_params)
        best_params["cutree"]["cutree-strength"] = strength_range[best_index]
        # best_params, best_cost = self.ternary_search(
        #     video_sequences, "cutree", param_manager, strength_range
        # )
        self.log(
            f"cutree module best cost: {best_cost}, best strength: {best_params['cutree']['cutree-strength']}"
        )
        return best_params

    def search_psyrdo(self, video_sequences, param_manager):
        current_params = param_manager.get_current_values()
        best_cost = self.cost_calculator.get_optimal_loss()
        best_params = copy.deepcopy(current_params)

        self.log("Testing psyrdo module when rd=1...")
        mode_1_params = copy.deepcopy(current_params)
        mode_1_params["psyrdo"]["rd"] = 1
        mode_1_cost = self.cost_calculator.calculate_cost(
            mode_1_params, video_sequences
        )
        if mode_1_cost < best_cost:
            best_cost = mode_1_cost
            best_params = mode_1_params

        self.log("Testing psyrdo module when rd=2...")
        mode_2_params = copy.deepcopy(current_params)
        mode_2_params["psyrdo"]["rd"] = 2
        mode_2_cost = self.cost_calculator.calculate_cost(
            mode_2_params, video_sequences
        )
        if mode_2_cost < best_cost:
            best_cost = mode_2_cost
            best_params = mode_2_params

        self.log("Testing psyrdo module when psydo is off...")
        off_params = copy.deepcopy(current_params)
        off_params["psyrdo"]["psy-rd"] = 0
        off_cost = self.cost_calculator.calculate_cost(off_params, video_sequences)
        if off_cost < best_cost:
            best_cost = off_cost
            best_params = off_params

        mode_range = [3, 5]
        strength_range = [float(Decimal(i) / Decimal(10)) for i in range(1, 51)]
        max_iterations = 100
        self.log("Testing psyrdo module when psydo is open...")
        best_open_params, best_open_cost = self.coordinate_descent(
            video_sequences,
            "psyrdo",
            param_manager,
            max_iterations,
            mode_range,
            strength_range,
        )
        if best_open_cost < best_cost:
            best_cost = best_open_cost
            best_params = best_open_params

        self.log(
            f"psyrdo module best cost: {best_cost}, best mode: {best_params['psyrdo']['rd']}, best strength: {best_params['psyrdo']['psy-rd']}"
        )
        return best_params

    def search_psyrdoq(self, video_sequences, param_manager):
        current_params = param_manager.get_current_values()
        best_cost = self.cost_calculator.get_optimal_loss()
        best_params = copy.deepcopy(current_params)

        self.log("Testing psyrdoq module when rdoq is off...")
        off_params = copy.deepcopy(current_params)
        off_params["psyrdoq"]["rdoq-level"] = 0
        off_cost = self.cost_calculator.calculate_cost(off_params, video_sequences)
        if off_cost < best_cost:
            best_cost = off_cost
            best_params = off_params

        mode_range = [1, 2]
        strength_range = [float(Decimal(i) / Decimal(10)) for i in range(1, 101)] + [
            i for i in range(11, 51)
        ]
        max_iterations = 280
        self.log("Testing psyrdoq module when rdoq is open...")
        best_open_params, best_open_cost = self.coordinate_descent(
            video_sequences,
            "psyrdoq",
            param_manager,
            max_iterations,
            mode_range,
            strength_range,
        )
        if best_open_cost < best_cost:
            best_cost = best_open_cost
            best_params = best_open_params

        self.log(
            f"psyrdoq module best cost: {best_cost}, best mode: {best_params['psyrdoq']['rdoq-level']}, best strength: {best_params['psyrdoq']['psy-rdoq']}"
        )
        return best_params

    def search_qcomp(self, video_sequences, param_manager):
        self.log("Testing qcomp module...")
        current_params = param_manager.get_current_values()
        strength_range = [float(Decimal(i) / Decimal(100)) for i in range(50, 101)]
        initial_cost = self.cost_calculator.get_optimal_loss()
        initial_index = strength_range.index(current_params["qcomp"]["qcomp"])
        best_index, best_cost = self.simulated_annealing(
            video_sequences,
            "qcomp",
            param_manager,
            strength_range,
            initial_index,
            initial_cost,
        )
        # best_params, best_cost = self.ternary_search(
        #     video_sequences, "qcomp", param_manager, strength_range
        # )
        best_params = copy.deepcopy(current_params)
        best_params["qcomp"]["qcomp"] = strength_range[best_index]
        self.log(
            f"qcomp module best cost: {best_cost}, best strength: {best_params['qcomp']['qcomp']}"
        )
        return best_params

    def coordinate_descent(
        self,
        video_sequences,
        module_name,
        param_manager,
        max_iterations,
        mode_range,
        strength_range,
    ):
        self.log(f"Starting coordinate descent for module: {module_name}")
        iter_count = 0
        current_params = param_manager.get_current_values()

        module_params = current_params[module_name]
        param_names = list(module_params.keys())
        mode_param_name = param_names[0]
        strength_param_name = param_names[1]

        while iter_count < max_iterations:
            prev_params = copy.deepcopy(current_params)
            current_mode = current_params[module_name][mode_param_name]
            self.log(
                f"iter_count: {iter_count} module: {module_name} current_mode: {current_mode} current_strength: {current_params[module_name][strength_param_name]}"
            )
            current_strength_index = strength_range.index(
                current_params[module_name][strength_param_name]
            )

            left_index = max(current_strength_index - 1, 0)
            right_index = min(current_strength_index + 1, len(strength_range) - 1)

            current_cost = self.cost_calculator.calculate_cost(
                current_params, video_sequences
            )
            left_params = copy.deepcopy(current_params)
            left_params[module_name][strength_param_name] = strength_range[left_index]
            left_cost = self.cost_calculator.calculate_cost(
                left_params, video_sequences
            )

            right_params = copy.deepcopy(current_params)
            right_params[module_name][strength_param_name] = strength_range[right_index]
            right_cost = self.cost_calculator.calculate_cost(
                right_params, video_sequences
            )

            if left_cost < current_cost and left_cost < right_cost:
                search_direction = -1
            elif right_cost < current_cost and right_cost < left_cost:
                search_direction = 1
            else:
                search_direction = 0

            best_strength_index = current_strength_index
            best_cost = current_cost
            new_index = current_strength_index + search_direction
            while 0 <= new_index < len(strength_range):
                new_params = copy.deepcopy(current_params)
                new_params[module_name][strength_param_name] = strength_range[new_index]
                new_cost = self.cost_calculator.calculate_cost(
                    new_params, video_sequences
                )
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_strength_index = new_index
                    new_index += search_direction
                else:
                    break

            current_params[module_name][strength_param_name] = strength_range[
                best_strength_index
            ]

            best_mode = current_mode
            best_cost = self.cost_calculator.calculate_cost(
                current_params, video_sequences
            )
            for mode in mode_range:
                new_params = copy.deepcopy(current_params)
                new_params[module_name][mode_param_name] = mode
                new_cost = self.cost_calculator.calculate_cost(
                    new_params, video_sequences
                )
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mode = mode

            current_params[module_name][mode_param_name] = best_mode

            if all(
                prev_params[module_name][key] == current_params[module_name][key]
                for key in prev_params[module_name]
            ):
                break

            iter_count += 1

        optimal_cost = self.cost_calculator.calculate_cost(
            current_params, video_sequences
        )
        return current_params, optimal_cost

    def ternary_search(
        self, video_sequences, module_name, param_manager, strength_range
    ):
        self.log(f"Starting ternary search for module: {module_name}")
        current_params = param_manager.get_current_values()

        module_params = current_params.get(module_name)
        if module_name == "cutree":
            strength_param_name = "cutree-strength"
        elif module_name == "qcomp":
            strength_param_name = "qcomp"

        initial_strength = module_params.get(strength_param_name)
        initial_index = strength_range.index(initial_strength)

        current_index = initial_index
        current_cost = self.cost_calculator.calculate_cost(
            current_params, video_sequences
        )

        right_index = min(initial_index + 1, len(strength_range) - 1)
        right_params = copy.deepcopy(current_params)
        right_params[module_name][strength_param_name] = strength_range[right_index]
        right_cost = self.cost_calculator.calculate_cost(right_params, video_sequences)

        left_index = max(current_index - 1, 0)
        left_params = copy.deepcopy(current_params)
        left_params[module_name][strength_param_name] = strength_range[left_index]
        left_cost = self.cost_calculator.calculate_cost(left_params, video_sequences)

        if right_cost < current_cost and right_cost < left_cost:
            left_bound = current_index
            right_bound = len(strength_range) - 1
            last_direction = 1
        elif left_cost < current_cost and left_cost < right_cost:
            left_bound = 0
            right_bound = current_index
            last_direction = -1
        else:
            left_distance = current_index
            right_distance = len(strength_range) - 1 - current_index
            if left_distance > right_distance:
                left_bound = 0
                right_bound = current_index
                last_direction = -1
            else:
                left_bound = current_index
                right_bound = len(strength_range) - 1
                last_direction = 1

        iter_count = 0
        max_iterations = len(strength_range)
        best_index = current_index
        best_cost = current_cost
        non_convergence_count = 0
        performance_history = []

        while iter_count < max_iterations and (right_bound - left_bound) > 3:
            self.log(
                f"iter_count: {iter_count} left_bound: {left_bound} right_bound: {right_bound}"
            )
            left_third_index = left_bound + (right_bound - left_bound) // 3
            right_third_index = right_bound - (right_bound - left_bound) // 3

            left_params = copy.deepcopy(current_params)
            left_params[module_name][strength_param_name] = strength_range[
                left_third_index
            ]
            left_cost = self.cost_calculator.calculate_cost(
                left_params, video_sequences
            )

            right_params = copy.deepcopy(current_params)
            right_params[module_name][strength_param_name] = strength_range[
                right_third_index
            ]
            right_cost = self.cost_calculator.calculate_cost(
                right_params, video_sequences
            )

            performance_history.append((left_cost, right_cost))

            if left_cost < right_cost:
                right_bound = right_third_index
                if last_direction == -1:
                    non_convergence_count += 1
                last_direction = -1
            else:
                left_bound = left_third_index
                if last_direction == 1:
                    non_convergence_count += 1
                last_direction = 1

            if left_cost < best_cost:
                best_index = left_third_index
                best_cost = left_cost
            if right_cost < best_cost:
                best_index = right_third_index
                best_cost = right_cost

            iter_count += 1

            if len(performance_history) > 2:
                # 性能波动检测
                if (performance_history[-1][0] < performance_history[-1][1]) != (
                    performance_history[-2][0] < performance_history[-2][1]
                ):
                    self.log("performance unstable")
                    best_index_annealing, best_cost_annealing = (
                        self.simulated_annealing(
                            video_sequences,
                            module_name,
                            param_manager,
                            strength_range,
                            best_index,
                            best_cost,
                        )
                    )
                    if best_cost_annealing < best_cost:
                        best_index = best_index_annealing
                        best_cost = best_cost_annealing
                    best_params = copy.deepcopy(current_params)
                    best_params[module_name][strength_param_name] = strength_range[
                        best_index
                    ]
                    return best_params, best_cost
                # 多次无法收敛检测
                L = len(strength_range) - 1
                expected_iterations = math.ceil(math.log(L / 3, 3 / 2))
                if iter_count > expected_iterations and (right_bound - left_bound) > 3:
                    self.log("convergence failed")
                    best_index_annealing, best_cost_annealing = (
                        self.simulated_annealing(
                            video_sequences,
                            module_name,
                            param_manager,
                            strength_range,
                            best_index,
                            best_cost,
                        )
                    )
                    if best_cost_annealing < best_cost:
                        best_index = best_index_annealing
                        best_cost = best_cost_annealing
                    best_params = copy.deepcopy(current_params)
                    best_params[module_name][strength_param_name] = strength_range[
                        best_index
                    ]
                    return best_params, best_cost
                # 性能值非单调变化检测
                if (
                    performance_history[-1][1] < performance_history[-1][0]
                    and performance_history[-1][1] < performance_history[-2][1]
                ):
                    self.log("performance not monotonic")
                    best_index_annealing, best_cost_annealing = (
                        self.simulated_annealing(
                            video_sequences,
                            module_name,
                            param_manager,
                            strength_range,
                            best_index,
                            best_cost,
                        )
                    )
                    if best_cost_annealing < best_cost:
                        best_index = best_index_annealing
                        best_cost = best_cost_annealing
                    best_params = copy.deepcopy(current_params)
                    best_params[module_name][strength_param_name] = strength_range[
                        best_index
                    ]
                    return best_params, best_cost

        for index in range(left_bound, right_bound + 1):
            params = copy.deepcopy(current_params)
            params[module_name][strength_param_name] = strength_range[index]
            self.log(
                f"searching {module_name} {strength_param_name} {params[module_name][strength_param_name]}"
            )
            cost = self.cost_calculator.calculate_cost(params, video_sequences)
            if cost < best_cost:
                best_index = index
                best_cost = cost

        best_params = copy.deepcopy(current_params)
        best_params[module_name][strength_param_name] = strength_range[best_index]
        return best_params, best_cost

    def simulated_annealing(
        self,
        video_sequences,
        module_name,
        param_manager,
        strength_range,
        initial_index,
        initial_cost,
    ):
        self.log(f"start simulated annealing {module_name}")
        temperature = 100
        cooling_rate = 0.95

        current_params = param_manager.get_current_values()
        current_index = initial_index
        best_index = initial_index
        best_cost = initial_cost
        if module_name == "cutree":
            strength_param_name = "cutree-strength"
            perturbation_range = 7
        elif module_name == "qcomp":
            strength_param_name = "qcomp"
            perturbation_range = 10

        while temperature > 0.01:
            new_index = current_index + random.randint(
                -perturbation_range, perturbation_range
            )
            new_index = max(0, min(len(strength_range) - 1, new_index))
            new_params = copy.deepcopy(current_params)
            new_params[module_name][strength_param_name] = strength_range[new_index]
            self.log(
                f"simulated annealing {module_name} {strength_param_name} {new_params[module_name][strength_param_name]} temperature: {temperature}"
            )
            new_cost = self.cost_calculator.calculate_cost(new_params, video_sequences)

            if new_cost < best_cost:
                best_index = new_index
                best_cost = new_cost
            elif self.accept_new_solution(best_cost, new_cost, temperature):
                current_index = new_index

            temperature *= cooling_rate

        return best_index, best_cost

    def accept_new_solution(self, best_cost, new_cost, temperature):
        # 避免指数溢出或下溢
        max_exp_input = 709  # math.exp(709) 接近 float 最大值
        min_exp_input = -745  # math.exp(-745) 接近 float 最小值
        exp_input = (best_cost - new_cost) / temperature
        exp_input = min(exp_input, max_exp_input)
        exp_input = max(exp_input, min_exp_input)
        probability = math.exp(exp_input)
        return random.random() < probability

    def optimize(self, video_sequences):
        self.log("start optimize")
        # 存储不同清晰度下的最优参数组合
        optimal_params_per_quality = {}

        # 获取不同的清晰度级别
        quality_levels = list(video_sequences.keys())

        # 按模块优先级排序
        module_order = self.priority_sorter.get_ordered_modules()

        # 遍历不同的清晰度级别
        for quality in quality_levels:
            self.log(f"start optimize {quality}")
            self.cost_calculator.log(f"start optimize {quality}")
            # 初始化当前清晰度下的最优参数组合
            param_manager = copy.deepcopy(self.param_manager)
            current_best_params = param_manager.get_current_values()

            # 依次搜索每个模块的最优参数组合
            for module in module_order:
                if module == "vaq":
                    self.log(f"start optimize {quality} vaq")
                    self.cost_calculator.log(f"start optimize {quality} vaq")
                    current_best_params = self.search_vaq(
                        video_sequences[quality], param_manager
                    )
                    param_manager.set_param_value(
                        "vaq", "aq-mode", current_best_params["vaq"]["aq-mode"]
                    )
                    param_manager.set_param_value(
                        "vaq", "aq-strength", current_best_params["vaq"]["aq-strength"]
                    )
                elif module == "cutree":
                    self.log(f"start optimize {quality} cutree")
                    self.cost_calculator.log(f"start optimize {quality} cutree")
                    current_best_params = self.search_cutree(
                        video_sequences[quality], param_manager
                    )
                    param_manager.set_param_value(
                        "cutree", "cutree", current_best_params["cutree"]["cutree"]
                    )
                    param_manager.set_param_value(
                        "cutree",
                        "cutree-strength",
                        current_best_params["cutree"]["cutree-strength"],
                    )
                elif module == "psyrdo":
                    self.log(f"start optimize {quality} psyrdo")
                    self.cost_calculator.log(f"start optimize {quality} psyrdo")
                    current_best_params = self.search_psyrdo(
                        video_sequences[quality], param_manager
                    )
                    param_manager.set_param_value(
                        "psyrdo", "rd", current_best_params["psyrdo"]["rd"]
                    )
                    param_manager.set_param_value(
                        "psyrdo", "psy-rd", current_best_params["psyrdo"]["psy-rd"]
                    )
                elif module == "psyrdoq":
                    self.log(f"start optimize {quality} psyrdoq")
                    self.cost_calculator.log(f"start optimize {quality} psyrdoq")
                    current_best_params = self.search_psyrdoq(
                        video_sequences[quality], param_manager
                    )
                    param_manager.set_param_value(
                        "psyrdoq",
                        "rdoq-level",
                        current_best_params["psyrdoq"]["rdoq-level"],
                    )
                    param_manager.set_param_value(
                        "psyrdoq",
                        "psy-rdoq",
                        current_best_params["psyrdoq"]["psy-rdoq"],
                    )
                elif module == "qcomp":
                    self.log(f"start optimize {quality} qcomp")
                    self.cost_calculator.log(f"start optimize {quality} qcomp")
                    current_best_params = self.search_qcomp(
                        video_sequences[quality], param_manager
                    )
                    param_manager.set_param_value(
                        "qcomp", "qcomp", current_best_params["qcomp"]["qcomp"]
                    )

            optimal_params_per_quality[quality] = param_manager.get_current_values()
            self.cost_calculator.reset()

        self.log("finish optimize")
        return optimal_params_per_quality
