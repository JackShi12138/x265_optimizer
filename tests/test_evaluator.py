import sys
import os
import unittest
import shutil
from pathlib import Path

# 确保能导入 core_refactor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_refactor.cost import X265CostEvaluator
from core_refactor.params import SearchSpace

# === 配置 ===
# 请确认此路径无误
X265_PATH = "/home/shiyushen/program/x265_4.0/x265_release"
# 调试模式下，我们使用固定路径，方便你去文件夹里查看
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "debug_output")


class TestDebug(unittest.TestCase):
    def setUp(self):
        print(f"\n[Setup] 创建临时目录: {TEST_OUTPUT_DIR}")
        if not os.path.exists(TEST_OUTPUT_DIR):
            os.makedirs(TEST_OUTPUT_DIR)

        self.width = 64
        self.height = 64
        self.fps = 24
        self.video_name = f"TestVideo_{self.width}x{self.height}_{self.fps}.yuv"
        self.video_path = os.path.join(TEST_OUTPUT_DIR, self.video_name)

        # === 修改点 1: 生成更长的视频 (60帧) ===
        # 避免 x265 slow preset 因为帧数太少(Lookahead)导致不出数据
        print(f"[Setup] 生成测试视频: {self.video_path} (60 frames)")
        y_size = self.width * self.height
        uv_size = y_size // 4
        num_frames = 60

        try:
            with open(self.video_path, "wb") as f:
                for _ in range(num_frames):
                    f.write(bytes([0] * y_size))  # Y (Black)
                    f.write(bytes([128] * uv_size))  # U (Neutral)
                    f.write(bytes([128] * uv_size))  # V (Neutral)
            print("[Setup] 视频生成成功！")
        except Exception as e:
            self.fail(f"视频生成失败: {e}")

        # 再次确认文件存在
        if not os.path.exists(self.video_path):
            self.fail("致命错误：视频文件写入后无法找到！")

        self.evaluator = X265CostEvaluator(
            base_path=TEST_OUTPUT_DIR, x265_path=X265_PATH
        )
        self.space = SearchSpace()
        self.default_params = self.space.get_all_config()

    def tearDown(self):
        # === 修改点 2: 不删除文件 ===
        print(
            f"\n[Teardown]以此保留测试文件，请在: {TEST_OUTPUT_DIR} 中检查生成的 .yuv 和 .csv 文件"
        )
        # if os.path.exists(TEST_OUTPUT_DIR):
        #     shutil.rmtree(TEST_OUTPUT_DIR)

    def test_run(self):
        print(f"\n[Run] 开始调用 Evaluate...")

        # 构造输入
        test_sequences = {self.video_path: 500}

        try:
            cost = self.evaluator.evaluate(self.default_params, test_sequences)
            print(f"[Run] Evaluate 完成. Cost: {cost}")

            # 检查 CSV 是否存在
            expected_csv = os.path.join(
                TEST_OUTPUT_DIR, f"TestVideo_{self.width}x{self.height}_{self.fps}.csv"
            )
            if os.path.exists(expected_csv):
                print(f"[Check] CSV 文件已生成: {expected_csv}")
                # 打印 CSV 内容的前几行
                with open(expected_csv, "r") as f:
                    print("--- CSV Content Head ---")
                    for _ in range(3):
                        print(f.readline().strip())
                    print("------------------------")
            else:
                print(f"[Check] !!! CSV 文件未生成 !!!")

            self.assertIsNotNone(cost)
            self.assertNotEqual(cost, float("inf"))

        except Exception as e:
            print(f"[Run] 发生异常: {e}")
            self.fail(f"测试失败: {e}")


if __name__ == "__main__":
    unittest.main()
