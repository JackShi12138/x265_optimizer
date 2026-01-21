import concurrent.futures
from multiprocessing import Pool
import os
import subprocess
import pandas as pd
import math
import time
import json
import logging


def find_column_index(df, target_string):
    """查找包含目标字符串的列索引"""
    for col_index, col_name in enumerate(df.columns):
        if target_string in col_name:
            return col_index
    return None


def get_column_values(df, column_index):
    """获取指定列的非空数值"""
    if column_index is None:
        return []
    values = []
    for row_index in range(len(df)):
        value = df.iat[row_index, column_index]
        if pd.isna(value):
            break
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def calculate_average(values):
    """计算数值列表的平均值"""
    return sum(values) / len(values) if values else None


class CostCalculator:
    def __init__(self):
        self.log_file = open("cost_calculator_log.txt", "w", encoding="utf-8")
        # 用于存储已搜索参数及其对应的平均率失真损失的表
        self.parameter_table = {}
        # 存储全局最优的平均率失真损失
        self.global_optimal_rd_loss = float("inf")
        self.base_path = r"/home/shiyushen/result/"
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

    def calculate_cost(self, params, video_sequences):
        hashable_params = frozenset(
            (k, frozenset(v.items())) for k, v in params.items()
        )
        if hashable_params in self.parameter_table:
            return self.parameter_table[hashable_params]
        avg_rd_loss = self._parallel_calculate_rd_loss(params, video_sequences)
        self.parameter_table[hashable_params] = avg_rd_loss
        # 更新全局最优的平均率失真损失
        if avg_rd_loss < self.global_optimal_rd_loss:
            self.global_optimal_rd_loss = avg_rd_loss
        self.log(
            f"参数: {params}, 平均率失真损失: {avg_rd_loss}, 全局最优: {self.global_optimal_rd_loss}"
        )
        return avg_rd_loss

    def _parallel_calculate_rd_loss(self, params, video_sequences):
        """
        并行计算率失真损失
        1. 将22个视频分为两组，每组11个
        2. 顺序运行两组，组内并行运行11个x265
        3. 若某组平均率失真损失过高（超过全局最优平均率失真损失的2倍），则跳过该参数下一组的计算
        4. 计算所有组的平均率失真损失并返回
        """
        videos = list(video_sequences.keys())
        group1 = videos[:11]
        group2 = videos[11:]
        total_rd_loss = 0
        group_count = 0

        for video_group in [group1, group2]:
            group_rd_loss = self._calculate_group_rd_loss(
                params, video_group, video_sequences
            )
            if group_rd_loss is not None:
                if group_rd_loss > 2 * self.global_optimal_rd_loss:
                    continue
                total_rd_loss += group_rd_loss
                group_count += 1

            self._cleanup_files(video_group)

        if group_count == 0:
            return float("inf")
        return total_rd_loss / group_count

    def _calculate_group_rd_loss(self, params, video_group, video_sequences):
        """
        计算一组视频的平均率失真损失
        1. 并行对组内每个视频调用单视频计算函数
        2. 计算该组的平均率失真损失并返回
        """
        total_rd_loss = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._calculate_single_video_rd_loss,
                    params,
                    video,
                    video_sequences[video],
                )
                for video in video_group
            ]
            for future in concurrent.futures.as_completed(futures):
                rd_loss = future.result()
                total_rd_loss += rd_loss
        return total_rd_loss / len(video_group)

    def _calculate_single_video_rd_loss(self, params, video, bitrate):
        """
        计算单个视频的率失真损失
        1. 运行x265，保存输出日志为csv文件
        2. 读取csv文件，计算平均码率和平均qp
        3. 通过平均qp计算率失真损失用的λ
        4. 调用ffmpeg将x265输出的hevc文件转码为yuv
        5. 调用vmaf计算编码后视频的vmaf值，保存输出至json文件
        6. 读取json文件中的平均vmaf，计算失真为100 - vmaf
        7. 通过失真、λ和平均码率计算率失真损失并返回
        """
        x265_params = self._extract_x265_params(params)
        hevc_file, csv_file = self._run_x265_and_get_csv(x265_params, video, bitrate)
        rd_loss = self._read_csv_and_calculate(csv_file, video)
        resolution, fps = self.extract_resolution_and_fps(video)
        width, height = resolution.split("x")
        height = int(height)
        normalized_rd_loss = rd_loss / height
        return normalized_rd_loss

    def _extract_x265_params(self, params):
        """
        从 params 中提取真正需要传入 x265 的参数
        """
        x265_params = {}
        for module_params in params.values():
            x265_params.update(module_params)
        return x265_params

    def _run_x265_and_get_csv(self, x265_params, video, bitrate):
        """
        运行x265并保存输出日志为csv文件，返回csv文件路径
        """
        resolution, fps = self.extract_resolution_and_fps(video)
        cmd = [
            "/home/shiyushen/Release_4.0/build/linux/x265",
            "--input",
            video,
            "--input-res",
            resolution,
            "--fps",
            str(fps),
            "--bitrate",
            str(bitrate),
            "--strict-cbr",
            "--vbv-bufsize",
            "50000",
            "--vbv-maxrate",
            str(bitrate),
            "--csv-log-level",
            "2",
            "--preset",
            "slow",
        ]
        for k, v in x265_params.items():
            if k == "cutree":
                cmd.extend(["--cutree"])
            else:
                param_name = f"--{k}"
                cmd.extend([param_name, str(v)])
        video_name = self.extract_video_name(video)
        hevc_file = os.path.join(self.base_path, f"{video_name}.hevc")
        cmd.extend(["-o", hevc_file])
        csv_file = os.path.join(self.base_path, f"{video_name}.csv")
        cmd.extend(["--csv", csv_file])

        try:
            subprocess.run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"x265 运行出错: {e}")
        return hevc_file, csv_file

    def _read_csv_and_calculate(self, csv_file, video):
        resolution, fps = self.extract_resolution_and_fps(video)
        df = pd.read_csv(csv_file)
        # 查找各列索引并获取对应值和平均值
        columns = ["QP", "Bits", "Avg Luma Distortion", "Avg Chroma Distortion"]
        average_values = {}
        for column in columns:
            column_index = find_column_index(df, column)
            if column_index is None:
                print(f"未找到包含 {column} 的列")
                average_values[column] = None
            else:
                values = get_column_values(df, column_index)
                average_values[column] = calculate_average(values)

        # 计算 lamda、total_ctu、distortion 和 cost
        average_qp = average_values["QP"]
        average_bits = average_values["Bits"]
        average_luma_distortion = average_values["Avg Luma Distortion"]
        average_chroma_distortion = average_values["Avg Chroma Distortion"]

        if average_qp is not None:
            lamda = 0.038 * math.exp(0.234 * average_qp)
        else:
            lamda = None

        width, height = resolution.split("x")
        width = int(width)
        height = int(height)
        total_ctu = math.ceil(width / 64) * math.ceil(height / 64)

        if (
            average_luma_distortion is not None
            and average_chroma_distortion is not None
        ):
            distortion = (
                average_luma_distortion + average_chroma_distortion
            ) * total_ctu
        else:
            distortion = None

        if lamda is not None and average_bits is not None:
            cost = distortion + lamda * average_bits if distortion is not None else None
        else:
            cost = None

        return cost

    def extract_resolution_and_fps(self, video_path):
        # 从路径中提取文件名
        file_name = video_path.split("/")[-1]
        # 按 _ 分割文件名
        parts = file_name.split("_")
        if len(parts) >= 3:
            resolution = parts[1]
            try:
                fps = int(parts[2].split(".")[0])  # 去除可能的文件扩展名
                return resolution, fps
            except ValueError:
                print("无法将帧率转换为整数。")
        else:
            print("文件名格式不符合要求，无法提取分辨率和帧率。")
        return None, None

    def _cleanup_files(self, video_group):
        """
        清理调用各个程序所产生的文件
        """
        for video in video_group:
            # 构建各个文件的路径
            video_name = self.extract_video_name(video)
            csv_file = os.path.join(self.base_path, f"{video_name}.csv")
            hevc_file = os.path.join(self.base_path, f"{video_name}.hevc")
            # 删除文件
            for file in [csv_file, hevc_file]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                        print(f"已删除文件: {file}")
                    except OSError as e:
                        print(f"删除文件 {file} 时出错: {e}")

    def extract_video_name(self, video_path):
        # 从路径中提取文件名（包含扩展名）
        file_name = os.path.basename(video_path)
        # 去除文件名中的扩展名
        base_name = os.path.splitext(file_name)[0]
        # 查找第一个下划线的位置
        underscore_index = base_name.find("_")
        if underscore_index != -1:
            # 如果找到下划线，提取下划线之前的部分作为视频名
            video_name = base_name[:underscore_index]
        else:
            # 如果没有找到下划线，整个文件名作为视频名
            video_name = base_name
        return video_name

    def reset(self):
        self.global_optimal_rd_loss = float("inf")
        self.parameter_table = {}
        self.log("reset successfully")

    def get_optimal_loss(self):
        return self.global_optimal_rd_loss
