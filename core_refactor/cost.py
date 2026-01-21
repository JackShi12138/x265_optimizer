import concurrent.futures
import os
import subprocess
import pandas as pd
import math
import time
from .base import CostEvaluator


class X265CostEvaluator(CostEvaluator):
    def __init__(
        self,
        base_path="/home/shiyushen/result/",
        x265_path="/home/shiyushen/program/x265_4.0/x265_release",
    ):
        self.base_path = base_path
        self.x265_path = x265_path
        self.cache = {}
        self.global_min_cost = float("inf")
        # 使用追加模式 'a'，防止多进程或多次运行时覆盖日志，也可以改回 'w'
        self.log_file = open("cost_log_refactor.txt", "w", encoding="utf-8")

    def reset(self):
        self.cache = {}
        self.global_min_cost = float("inf")

    def evaluate(self, params: dict, video_sequences: dict = None) -> float:
        param_key = frozenset((m, frozenset(p.items())) for m, p in params.items())

        if param_key in self.cache:
            return self.cache[param_key]

        if video_sequences is None:
            raise ValueError("First evaluation requires video_sequences")

        cost = self._parallel_calculate_rd_loss(params, video_sequences)
        self.cache[param_key] = cost

        if cost < self.global_min_cost:
            self.global_min_cost = cost

        self._log(f"Params: {params} -> Cost: {cost}")
        return cost

    def _log(self, msg):
        try:
            self.log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.log_file.flush()
        except ValueError:
            pass  # 防止文件已关闭

    def _parallel_calculate_rd_loss(self, params, video_sequences):
        videos = list(video_sequences.keys())
        # 简单分组，防止一次性并发太多（虽然这里还是用group内并发）
        # 你的逻辑是 22 个视频分两组
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
        # 限制线程数，防止卡死机器，通常设为 CPU 核心数或更少
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
        """单视频执行与指标提取"""

        # --- 1. 修复文件名解析逻辑 ---
        # 必须先提取文件名，不能直接 split 路径
        filename = os.path.basename(video_path)  # e.g. TestVideo_64x64_24.yuv
        parts = filename.split("_")

        if len(parts) < 3:
            print(f"Error: Filename format incorrect: {filename}")
            return None

        # parts[0]: Name, parts[1]: Resolution, parts[2]: Fps.yuv
        res_str = parts[1]
        try:
            fps = parts[2].split(".")[0]
        except IndexError:
            print(f"Error: FPS parse failed for: {filename}")
            return None

        # --- 2. 构建命令 ---
        video_name_no_ext = os.path.splitext(filename)[0]
        csv_file = os.path.join(self.base_path, f"{video_name_no_ext}.csv")

        # 确保清理旧文件
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
            str(int(bitrate) * 2),  # 通常 bufsize 是 bitrate 的 1-2 倍
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

        # --- 3. 参数拼接 ---
        # 展平参数
        flat_params = {}
        for m in params.values():
            flat_params.update(m)

        for k, v in flat_params.items():
            if k == "cutree":
                if int(v) == 1:
                    cmd.append("--cutree")
                else:
                    cmd.append("--no-cutree")
            # 注意：如果你的 x265 不支持 --cutree-strength，请注释掉下面两行
            # 或者将其映射到正确的参数
            elif k == "cutree-strength":
                cmd.extend(["--cutree-strength", str(v)])
            else:
                cmd.extend([f"--{k}", str(v)])

        # --- 4. 运行 ---
        # Debug 模式下开启 stderr 输出
        print(f"DEBUG Executing: {' '.join(cmd)}")

        try:
            # 捕获 stderr 以便出错时打印，但平时保持安静
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                print(f"\n[x265 Error] Return Code {result.returncode}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Error Output: {result.stderr}\n")
                return None

        except Exception as e:
            print(f"Subprocess failed: {e}")
            return None

        # --- 5. 计算 Cost ---
        if not os.path.exists(csv_file):
            print(f"Error: CSV not generated for {filename}")
            return None

        return self._compute_normalized_cost(csv_file, res_str)

    def _compute_normalized_cost(self, csv_file, resolution):
        try:
            # 修改 1: 增加 skipinitialspace=True 自动处理 CSV 中的空格
            df = pd.read_csv(csv_file, skipinitialspace=True)
            
            # --- Debug: 打印 CSV 状态，排查由空文件或解析错误导致的问题 ---
            # print(f"\n[Debug CSV] File: {csv_file}")
            # print(f"Columns found: {list(df.columns)}")
            # if len(df) == 0:
            #    print("[Error] CSV is empty (contains no data rows)!")
            #    return None
            # print(f"First row: {df.iloc[0].to_dict() if len(df)>0 else 'No Data'}")
            # -----------------------------------------------------------

            def get_col(name):
                # 优先精确匹配
                if name in df.columns: return name
                # 其次模糊匹配
                for c in df.columns:
                    if name in c: return c
                return None

            qp_col = get_col("QP")
            bits_col = get_col("Bits")
            # x265 CSV 中通常叫 "Avg Luma Distortion"
            luma_col = get_col("Avg Luma Distortion") or get_col("Luma Distortion")
            chroma_col = get_col("Avg Chroma Distortion") or get_col("Chroma Distortion")
            
            if not all([qp_col, bits_col, luma_col, chroma_col]):
                print(f"Error: Missing columns in CSV {csv_file}")
                # 打印出当前有哪些列，方便调试
                print(f"Existing columns: {list(df.columns)}")
                return None

            # 强制转换为数值类型，防止因特殊字符导致被识别为 Object
            avg_qp = pd.to_numeric(df[qp_col], errors='coerce').mean()
            avg_bits = pd.to_numeric(df[bits_col], errors='coerce').mean()
            dist_luma = pd.to_numeric(df[luma_col], errors='coerce').mean()
            dist_chroma = pd.to_numeric(df[chroma_col], errors='coerce').mean()
            
            # 如果转换后全是 NaN (说明数据有问题)，或者文件本身为空
            if pd.isna(avg_qp) or pd.isna(avg_bits): 
                print(f"Error: Invalid data in CSV (NaN values). QP={avg_qp}, Bits={avg_bits}")
                return None
            
            lamda = 0.038 * math.exp(0.234 * avg_qp)
            height = int(resolution.split('x')[1])
            width = int(resolution.split('x')[0])
            # 计算 CTU 总数，用于将 Avg Distortion 转换为 Total Distortion
            total_ctu = math.ceil(width / 64) * math.ceil(height / 64)
            
            distortion = (dist_luma + dist_chroma) * total_ctu
            cost = distortion + lamda * avg_bits
            
            return cost / height 
            
        except Exception as e:
            print(f"Error calculating cost for {csv_file}: {e}")
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
