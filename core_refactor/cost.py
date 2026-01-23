import concurrent.futures
import os
import subprocess
import pandas as pd
import math
import time
import csv  # [新增] 引入csv模块
from .base import CostEvaluator


class X265CostEvaluator(CostEvaluator):
    def __init__(
        self,
        base_path="/home/shiyushen/result/",
        x265_path="/home/shiyushen/Release_4.0/build/linux/x265",
    ):
        self.base_path = base_path
        self.x265_path = x265_path
        self.cache = {}
        self.global_min_cost = float("inf")

        # 1. TXT 日志 (保持不变)
        self.log_file = open("cost_log_refactor.txt", "w", encoding="utf-8")

        # 2. [新增] CSV 日志 (用于画图分析)
        # 使用 newline='' 防止Windows下出现空行，Linux下无影响
        self.csv_file = open(
            "optimization_history.csv", "w", newline="", encoding="utf-8"
        )
        self.csv_writer = csv.writer(self.csv_file)
        # 写入表头
        self.csv_writer.writerow(
            [
                "Timestamp",
                "Quality",
                "Module",
                "Global_Iter",
                "Eval_Count",
                "Cost",
                "Params",
            ]
        )

        # 3. [新增] 评估计数器 (用于绘制 x轴: Evaluation Count)
        self.eval_count = 0

        # 上下文信息
        self.context_info = {"quality": "N/A", "module": "Init", "iter": 0}

    def reset(self):
        self.cache = {}
        self.global_min_cost = float("inf")
        # 重置计数器，确保每个清晰度的曲线都从 0 开始
        self.eval_count = 0
        self._log("Evaluator reset: Cache cleared, Min Cost & Counter reset.")

    def set_context(self, quality=None, module=None, iteration=None):
        """更新日志上下文信息"""
        if quality:
            self.context_info["quality"] = quality
        if module:
            self.context_info["module"] = module
        if iteration is not None:
            self.context_info["iter"] = iteration

    def evaluate(self, params: dict, video_sequences: dict = None) -> float:
        # 生成参数签名
        param_key = frozenset((m, frozenset(p.items())) for m, p in params.items())

        if param_key in self.cache:
            return self.cache[param_key]

        if video_sequences is None:
            raise ValueError("First evaluation requires video_sequences")

        # 增加计数
        self.eval_count += 1

        # 执行计算
        cost = self._parallel_calculate_rd_loss(params, video_sequences)
        self.cache[param_key] = cost

        is_new_best = False
        if cost < self.global_min_cost:
            self.global_min_cost = cost
            is_new_best = True

        self._log_evaluation(params, cost, is_new_best)
        return cost

    def _log(self, msg):
        try:
            self.log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.log_file.flush()
        except ValueError:
            pass

    def _log_evaluation(self, params, cost, is_best):
        """同时输出 TXT 和 CSV 日志"""
        timestamp = time.strftime("%H:%M:%S")
        ctx = self.context_info

        # --- 1. 写入 TXT (人类可读) ---
        marker = "★ NEW BEST" if is_best else ""
        log_str = (
            f"[{ctx['quality']}] "
            f"[Mod: {ctx['module']}] "
            f"[Iter: {ctx['iter']}] "
            f"[Eval: {self.eval_count}] "  # 加上计数方便核对
            f"Cost: {cost:.4f} {marker} "
            f"Params: {params}"
        )
        self._log(log_str)

        # --- 2. 写入 CSV (机器可读/画图用) ---
        try:
            self.csv_writer.writerow(
                [
                    timestamp,
                    ctx["quality"],
                    ctx["module"],
                    ctx["iter"],  # Global Iteration (外层循环)
                    self.eval_count,  # Inner Evaluation Count (绝对计数，适合做X轴)
                    cost,
                    str(params),  # 将字典转为字符串存储，读取时可用 eval() 还原
                ]
            )
            self.csv_file.flush()
        except Exception as e:
            print(f"CSV Write Error: {e}")

    def _parallel_calculate_rd_loss(self, params, video_sequences):
        videos = list(video_sequences.keys())
        group1, group2 = videos[:11], videos[11:]
        total_rd_loss = 0
        count = 0

        for group in [group1, group2]:
            if not group:
                continue
            # 剪枝逻辑
            if count > 0 and self.global_min_cost != float("inf"):
                current_avg = total_rd_loss / count
                if current_avg > 2 * self.global_min_cost:
                    return float("inf")

            group_loss = self._calculate_group_loss(params, group, video_sequences)
            if group_loss is None:
                return float("inf")

            total_rd_loss += group_loss
            count += 1
            self._cleanup(group)

        return total_rd_loss / count if count > 0 else float("inf")

    def _calculate_group_loss(self, params, group, sequences):
        group_total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
            futures = {
                executor.submit(self._run_single_video, params, v, sequences[v]): v
                for v in group
            }
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is None:
                    return None
                group_total += res
        return group_total / len(group)

    def _run_single_video(self, params, video_path, bitrate):
        filename = os.path.basename(video_path)
        parts = filename.split("_")
        if len(parts) < 3:
            return None
        res_str = parts[1]
        try:
            fps = parts[2].split(".")[0]
        except IndexError:
            return None

        video_name_no_ext = os.path.splitext(filename)[0]
        csv_file = os.path.join(self.base_path, f"{video_name_no_ext}.csv")
        if os.path.exists(csv_file):
            try:
                os.remove(csv_file)
            except OSError:
                pass

        cmd = [
            self.x265_path,
            "--input",
            video_path,
            "--input-res",
            res_str,
            "--fps",
            fps,
            "--bitrate",
            str(bitrate),
            "--strict-cbr",
            "--vbv-bufsize",
            str(int(bitrate) * 2),
            "--vbv-maxrate",
            str(bitrate),
            "--csv-log-level",
            "2",
            "--preset",
            "slow",
            "--csv",
            csv_file,
            "-o",
            "/dev/null",
        ]

        flat_params = {}
        for m in params.values():
            flat_params.update(m)
        for k, v in flat_params.items():
            if k == "cutree":
                if int(v) == 1:
                    cmd.append("--cutree")
                else:
                    cmd.append("--no-cutree")
            elif k == "cutree-strength":
                cmd.extend(["--cutree-strength", str(v)])
            else:
                cmd.extend([f"--{k}", str(v)])

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception:
            return None

        if not os.path.exists(csv_file):
            return None
        return self._compute_normalized_cost(csv_file, res_str)

    def _compute_normalized_cost(self, csv_file, resolution):
        try:
            df = pd.read_csv(csv_file, skipinitialspace=True)

            def get_col(name):
                if name in df.columns:
                    return name
                for c in df.columns:
                    if name in c:
                        return c
                return None

            qp_col, bits_col = get_col("QP"), get_col("Bits")
            enc_order_col = get_col("Encode Order") or get_col("EncodeOrder")
            luma_col = get_col("Avg Luma Distortion") or get_col("Luma Distortion")
            chroma_col = get_col("Avg Chroma Distortion") or get_col(
                "Chroma Distortion"
            )

            if not all([qp_col, bits_col, enc_order_col, luma_col, chroma_col]):
                return None

            # 数据清洗：基于 Encode Order 过滤/截断 Summary 行
            valid_order_mask = pd.to_numeric(
                df[enc_order_col], errors="coerce"
            ).notnull()
            if not valid_order_mask.all():
                first_invalid_idx = valid_order_mask.idxmin()
                if not valid_order_mask[first_invalid_idx]:
                    df = df.iloc[:first_invalid_idx]

            if len(df) == 0:
                return None

            avg_qp = pd.to_numeric(df[qp_col], errors="coerce").mean()
            avg_bits = pd.to_numeric(df[bits_col], errors="coerce").mean()
            dist_luma = pd.to_numeric(df[luma_col], errors="coerce").mean()
            dist_chroma = pd.to_numeric(df[chroma_col], errors="coerce").mean()

            if pd.isna(avg_qp) or pd.isna(avg_bits):
                return None

            lamda = 0.038 * math.exp(0.234 * avg_qp)
            height = int(resolution.split("x")[1])
            width = int(resolution.split("x")[0])
            total_ctu = math.ceil(width / 64) * math.ceil(height / 64)
            distortion = (dist_luma + dist_chroma) * total_ctu
            return (distortion + lamda * avg_bits) / height
        except Exception:
            return None

    def _cleanup(self, group):
        for v in group:
            filename = os.path.basename(v)
            name = os.path.splitext(filename)[0]
            csv = os.path.join(self.base_path, f"{name}.csv")
            if os.path.exists(csv):
                try:
                    os.remove(csv)
                except OSError:
                    pass
