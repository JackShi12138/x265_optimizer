import concurrent.futures
import os
import subprocess
import pandas as pd
import math
import time
import csv
import json
from .base import CostEvaluator


class X265CostEvaluator(CostEvaluator):
    def __init__(
        self,
        base_path="/home/shiyushen/result/",
        x265_path="/home/shiyushen/program/x265_4.0/x265_release",
        allow_pruning=True,
    ):
        """
        :param allow_pruning: 是否允许剪枝。
                              - True: 生产模式，遇到劣质解直接停止，返回 inf，速度快。
                              - False: 实验对比模式，强制跑完所有数据，返回真实 Cost，数据全。
        """
        self.base_path = base_path
        self.x265_path = x265_path
        self.allow_pruning = allow_pruning  # [新增] 开关
        self.cache = {}
        self.global_min_cost = float("inf")

        # 1. TXT 日志
        self.log_file = open("cost_log_refactor.txt", "a", encoding="utf-8")

        # 2. CSV 日志
        csv_filename = "optimization_history.csv"
        file_exists = os.path.exists(csv_filename)

        self.csv_file = open(csv_filename, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        if not file_exists:
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

        self.eval_count = 0
        self.context_info = {"quality": "N/A", "module": "Init", "iter": 0}

    def reset(self):
        self.cache = {}
        self.global_min_cost = float("inf")
        self.eval_count = 0
        self._log("Evaluator reset: Cache cleared, Min Cost & Counter reset.")

    def set_context(self, quality=None, module=None, iteration=None):
        if quality:
            self.context_info["quality"] = quality
        if module:
            self.context_info["module"] = module
        if iteration is not None:
            self.context_info["iter"] = iteration

    def evaluate(self, params: dict, video_sequences: dict = None) -> float:
        param_key = frozenset((m, frozenset(p.items())) for m, p in params.items())

        if param_key in self.cache:
            return self.cache[param_key]

        if video_sequences is None:
            raise ValueError("First evaluation requires video_sequences")

        self.eval_count += 1

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
        timestamp = time.strftime("%H:%M:%S")
        ctx = self.context_info

        marker = "★ NEW BEST" if is_best else ""
        log_str = (
            f"[{ctx['quality']}] "
            f"[Mod: {ctx['module']}] "
            f"[Iter: {ctx['iter']}] "
            f"[Eval: {self.eval_count}] "
            f"Cost: {cost:.4f} {marker} "
            f"Params: {params}"
        )
        self._log(log_str)

        try:
            self.csv_writer.writerow(
                [
                    timestamp,
                    ctx["quality"],
                    ctx["module"],
                    ctx["iter"],
                    self.eval_count,
                    cost,
                    str(params),
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

            # === [修改] 只有当允许剪枝时，才执行早停判断 ===
            if self.allow_pruning:
                if count > 0 and self.global_min_cost != float("inf"):
                    current_avg = total_rd_loss / count
                    if current_avg > 2 * self.global_min_cost:
                        # 剪枝生效，返回 inf
                        return float("inf")
            # ==========================================

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


class VMAFCostEvaluator(CostEvaluator):
    def __init__(
        self,
        base_path="/home/shiyushen/result/",
        x265_path="/home/shiyushen/program/x265_4.0/x265_release",
        vmaf_exec_path="vmaf",  # [修改] vmaf 可执行文件路径/命令
        allow_pruning=True,
    ):

        self.base_path = base_path
        self.x265_path = x265_path
        self.vmaf_exec_path = vmaf_exec_path
        self.allow_pruning = allow_pruning
        self.cache = {}
        self.global_min_cost = float("inf")
        self.eval_count = 0

        # 独立的日志文件
        self.log_file = open("cost_log_vmaf.txt", "a", encoding="utf-8")

        csv_filename = "optimization_history_vmaf.csv"
        file_exists = os.path.exists(csv_filename)
        self.csv_file = open(csv_filename, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(
                [
                    "Timestamp",
                    "Quality",
                    "Module",
                    "Iter",
                    "Eval",
                    "Cost",
                    "Bitrate",
                    "VMAF",
                    "Params",
                ]
            )

        self.context_info = {"quality": "N/A", "module": "Init", "iter": 0}

    def reset(self):
        self.cache = {}
        self.global_min_cost = float("inf")
        self.eval_count = 0
        self._log("Evaluator reset.")

    def set_context(self, quality=None, module=None, iteration=None):
        if quality:
            self.context_info["quality"] = quality
        if module:
            self.context_info["module"] = module
        if iteration is not None:
            self.context_info["iter"] = iteration

    def evaluate(self, params: dict, video_sequences: dict = None) -> float:
        param_key = frozenset((m, frozenset(p.items())) for m, p in params.items())
        if param_key in self.cache:
            return self.cache[param_key]
        if video_sequences is None:
            raise ValueError("First evaluation requires video_sequences")

        self.eval_count += 1

        results = self._parallel_calculate_vmaf_cost(params, video_sequences)
        cost = results["cost"]

        self.cache[param_key] = cost
        is_best = False
        if cost < self.global_min_cost:
            self.global_min_cost = cost
            is_best = True

        self._log_evaluation(params, results, is_best)
        return cost

    def _log(self, msg):
        try:
            self.log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.log_file.flush()
        except ValueError:
            pass

    def _log_evaluation(self, params, results, is_best):
        ctx = self.context_info
        marker = "★ NEW BEST" if is_best else ""
        log_str = (
            f"[{ctx['quality']}] [Mod: {ctx['module']}] [Eval: {self.eval_count}] "
            f"Cost: {results['cost']:.4f} {marker} | "
            f"VMAF: {results['vmaf']:.2f} | Bitrate: {results['bitrate']:.2f} kbps"
        )
        self._log(log_str)
        try:
            self.csv_writer.writerow(
                [
                    time.strftime("%H:%M:%S"),
                    ctx["quality"],
                    ctx["module"],
                    ctx["iter"],
                    self.eval_count,
                    results["cost"],
                    results["bitrate"],
                    results["vmaf"],
                    str(params),
                ]
            )
            self.csv_file.flush()
        except Exception:
            pass

    def _parallel_calculate_vmaf_cost(self, params, video_sequences):
        videos = list(video_sequences.keys())
        total_cost = 0
        total_vmaf = 0
        total_bitrate = 0
        count = 0

        # 并发计算 (VMAF 极其耗时，建议根据 CPU 核数调整 max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self._run_single_video_vmaf, params, v, video_sequences[v]
                ): v
                for v in videos
            }

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is None:
                    # 失败惩罚
                    return {"cost": float("inf"), "vmaf": 0, "bitrate": 0}

                total_cost += res["cost"]
                total_vmaf += res["vmaf"]
                total_bitrate += res["bitrate"]
                count += 1

        if count == 0:
            return {"cost": float("inf"), "vmaf": 0, "bitrate": 0}

        return {
            "cost": total_cost / count,
            "vmaf": total_vmaf / count,
            "bitrate": total_bitrate / count,
        }

    def _run_single_video_vmaf(self, params, video_path, target_bitrate):
        """
        运行单个视频：Encode -> VMAF Calc -> Cost Calc
        """
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]

        # 1. 解析文件名获取宽高等信息 (e.g. Video_1920x1080_30fps.yuv)
        parts = filename.split("_")
        if len(parts) < 3:
            return None
        res_str = parts[1]  # "1920x1080"
        try:
            width, height = map(int, res_str.split("x"))
            fps = parts[2].split(".")[0]
        except:
            return None

        # 定义输出路径
        recon_yuv = os.path.join(self.base_path, f"{name_no_ext}_recon.yuv")
        csv_file = os.path.join(self.base_path, f"{name_no_ext}.csv")
        vmaf_json = os.path.join(self.base_path, f"{name_no_ext}_vmaf.json")

        # 2. x265 编码 (生成 Recon YUV 和 CSV)
        cmd_x265 = [
            self.x265_path,
            "--input",
            video_path,
            "--input-res",
            res_str,
            "--fps",
            fps,
            "--bitrate",
            str(target_bitrate),
            "--strict-cbr",
            "--vbv-bufsize",
            str(int(target_bitrate) * 2),
            "--vbv-maxrate",
            str(target_bitrate),
            "--recon",
            recon_yuv,
            "--recon-depth",
            "8",  # 必须输出重构帧用于对比
            "--csv",
            csv_file,
            "--csv-log-level",
            "2",
            "-o",
            "/dev/null",
        ]

        # 参数注入
        flat_params = {}
        for m in params.values():
            flat_params.update(m)
        for k, v in flat_params.items():
            if k == "cutree":
                if int(v) == 1:
                    cmd_x265.append("--cutree")
                else:
                    cmd_x265.append("--no-cutree")
            elif k == "cutree-strength":
                cmd_x265.extend(["--cutree-strength", str(v)])
            else:
                cmd_x265.extend([f"--{k}", str(v)])

        try:
            subprocess.run(
                cmd_x265, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError:
            return None

        if not os.path.exists(recon_yuv) or not os.path.exists(csv_file):
            return None

        # 3. 读取 CSV 获取实际码率 (Real Bitrate)
        try:
            df = pd.read_csv(csv_file, skipinitialspace=True)
            # 简单清洗：找到 Bits 列
            bits_col = [c for c in df.columns if "Bits" in c][0]
            # 过滤非帧数据 (Summary 行)
            enc_order_col = [
                c for c in df.columns if "Encode Order" in c or "EncodeOrder" in c
            ][0]
            valid_mask = pd.to_numeric(df[enc_order_col], errors="coerce").notnull()
            if not valid_mask.all():
                df = df.iloc[: valid_mask.idxmin()]

            avg_bits = pd.to_numeric(df[bits_col]).mean()
            real_bitrate = (avg_bits * float(fps)) / 1000.0  # kbps
        except:
            # 降级策略
            real_bitrate = float(target_bitrate)

        # 4. 计算 VMAF (使用 vmaf 命令行工具)
        cmd_vmaf = [
            self.vmaf_exec_path,
            "-r",
            video_path,  # Reference
            "-d",
            recon_yuv,  # Distorted (Recon)
            "-w",
            str(width),
            "-h",
            str(height),
            "-p",
            "420",
            "-b",
            "8",
            "--json",
            "-o",
            vmaf_json,
        ]

        vmaf_score = 0
        try:
            subprocess.run(
                cmd_vmaf,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 解析 JSON
            with open(vmaf_json, "r") as f:
                vmaf_data = json.load(f)
                if (
                    "pooled_metrics" in vmaf_data
                    and "vmaf" in vmaf_data["pooled_metrics"]
                ):
                    vmaf_score = vmaf_data["pooled_metrics"]["vmaf"]["mean"]
                elif "VMAF score" in vmaf_data:
                    vmaf_score = vmaf_data["VMAF score"]
                elif "vmaf" in vmaf_data:  # 兼容某些旧版本
                    vmaf_score = vmaf_data["vmaf"]
                else:
                    print(f"Unknown VMAF JSON structure in {vmaf_json}")

        except Exception as e:
            # print(f"VMAF Failed: {e}")
            pass

        # 5. 清理大文件
        for f_path in [recon_yuv, csv_file, vmaf_json]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except:
                    pass

        if vmaf_score <= 0:
            return None

        # 6. 计算 Cost (Normalized)
        # Cost = (RealBitrate / TargetBitrate) * (100 / VMAF)^k
        k = 3
        bitrate_ratio = real_bitrate / float(target_bitrate)
        perceptual_cost = bitrate_ratio * pow(100.0 / vmaf_score, k)

        return {"cost": perceptual_cost, "vmaf": vmaf_score, "bitrate": real_bitrate}
