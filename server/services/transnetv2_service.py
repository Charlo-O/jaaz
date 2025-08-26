"""
TransNetV2 视频场景分割服务
用于视频处理和场景分割
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TransNetV2Service:
    """TransNetV2视频分割服务类"""

    def __init__(self):
        """初始化TransNetV2服务"""
        self.model = None
        self.model_type = None  # 'pytorch' 或 None
        self.transnet_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "TransNetV2"
        )
        self._initialize_model()

<<<<<<< Updated upstream
=======
    def _filter_base64_from_data(self, data):
        """过滤数据中的 base64 编码，避免在终端中显示

        这个函数与图片提供商的过滤机制保持一致，
        确保视频处理时也不会在终端显示base64数据
        """
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                # 只过滤超过10MB的巨大数据
                if isinstance(value, str) and len(value) > 10 * 1024 * 1024:  # 10MB
                    filtered[key] = f"[filtered video data - {len(value)} chars]"
                elif isinstance(value, (dict, list)):
                    filtered[key] = self._filter_base64_from_data(value)
                else:
                    filtered[key] = value
            return filtered
        elif isinstance(data, list):
            return [self._filter_base64_from_data(item) for item in data]
        else:
            return data

>>>>>>> Stashed changes
    def _initialize_model(self):
        """初始化TransNetV2模型"""
        print("🔍 开始初始化TransNetV2模型...")

        # 检查依赖
        dependencies: Dict[str, Optional[str]] = {
            'torch': None,
            'numpy': None,
            'cv2': None,
            'ffmpeg': None,
        }

        for dep in dependencies:
            try:
                if dep == 'cv2':
                    import cv2

                    dependencies[dep] = cv2.__version__
                elif dep == 'torch':
                    import torch

                    dependencies[dep] = torch.__version__
                elif dep == 'numpy':
                    import numpy as np

                    dependencies[dep] = np.__version__
                else:
                    __import__(dep)
                    dependencies[dep] = "installed"
                print(f"✅ {dep}: {dependencies[dep]}")
            except ImportError:
                print(f"❌ {dep}: 未安装")
                dependencies[dep] = None

        try:
            # 尝试使用PyTorch版本
            if dependencies['torch']:
                print("🔍 尝试加载PyTorch版本...")
                pytorch_path = os.path.join(self.transnet_path, "inference-pytorch")
                weights_path = os.path.join(
                    pytorch_path,
                    "transnetv2-pytorch-weights",
                    "transnetv2-pytorch-weights.pth",
                )

                print(f"📁 检查路径: {pytorch_path}")
                print(f"📁 权重文件: {weights_path}")

                if os.path.exists(pytorch_path):
                    if os.path.exists(weights_path):
                        sys.path.insert(0, pytorch_path)
                        try:
                            import torch

                            print(f"📦 PyTorch版本: {torch.__version__}")

                            from transnetv2_pytorch import (
                                TransNetV2 as TransNetV2PyTorch,
                            )

                            print("📦 导入TransNetV2PyTorch类成功")

                            self.model = TransNetV2PyTorch()
                            state_dict = torch.load(weights_path, map_location='cpu')
                            self.model.load_state_dict(state_dict)
                            self.model.eval()
                            self.model_type = 'pytorch'
                            print("✅ TransNetV2 PyTorch模型加载成功!")
                            logger.info("TransNetV2 PyTorch模型加载成功")
                            return
                        except Exception as e:
                            print(f"❌ PyTorch版本加载失败: {e}")
                            logger.warning(f"PyTorch版本加载失败: {e}")
                    else:
                        print("💡 PyTorch权重文件不存在，请检查模型安装")
                        print(f"❌ 需要权重文件: {weights_path}")
                else:
                    print(f"❌ PyTorch路径不存在: {pytorch_path}")

            print("⚠️ 无法加载TransNetV2模型，将使用基本的视频处理方法")
            logger.warning("无法加载TransNetV2模型，将使用基本的视频处理方法")

        except Exception as e:
            print(f"❌ 初始化TransNetV2模型时发生错误: {e}")
            logger.error(f"初始化TransNetV2模型时发生错误: {e}")

    def extract_key_frames(
        self,
        video_path: str,
        scenes: List[Tuple[int, int]],
        max_frames_per_scene: int = 1,
    ) -> List[np.ndarray]:
        """
        从视频的每个场景中提取关键帧

        Args:
            video_path: 视频文件路径
            scenes: 场景列表，每个元素为(开始帧, 结束帧)
            max_frames_per_scene: 每个场景最多提取的帧数

        Returns:
            key_frames: 关键帧列表
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            key_frames = []

            for scene_start, scene_end in scenes:
                # 对于每个场景，选择中间帧作为关键帧
                if max_frames_per_scene == 1:
                    # 选择场景中间的帧
                    key_frame_idx = (scene_start + scene_end) // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # 转换BGR到RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        key_frames.append(frame_rgb)
                else:
                    # 在场景中均匀选择多个帧
                    scene_length = scene_end - scene_start + 1
                    step = max(1, scene_length // max_frames_per_scene)

                    for i in range(max_frames_per_scene):
                        frame_idx = scene_start + i * step
                        if frame_idx <= scene_end:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            if ret:
                                # 转换BGR到RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                key_frames.append(frame_rgb)

            cap.release()
            print(f"🎬 从 {len(scenes)} 个场景中提取了 {len(key_frames)} 个关键帧")
            return key_frames

        except Exception as e:
            logger.error(f"提取关键帧时发生错误: {e}")
            raise

    def extract_video_frames(
        self, video_path: str, target_size: Tuple[int, int] = (48, 27)
    ) -> np.ndarray:
        """
        从视频中提取帧并调整大小

        Args:
            video_path: 视频文件路径
            target_size: 目标尺寸 (width, height)

        Returns:
            frames: shape为 [frames, height, width, 3] 的numpy数组
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 调整尺寸
                frame_resized = cv2.resize(frame_rgb, target_size)
                frames.append(frame_resized)

            cap.release()

            if not frames:
                raise ValueError("视频中没有找到帧")

            return np.array(frames, dtype=np.uint8)

        except Exception as e:
            logger.error(f"提取视频帧时发生错误: {e}")
            raise

    def predict_transitions(
        self, video_path: str, threshold: float = 0.5
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        预测视频中的场景转换

        Args:
            video_path: 视频文件路径
            threshold: 检测阈值

        Returns:
            predictions: 每一帧的转换概率
            scenes: 场景段列表，每个元素为(开始帧, 结束帧)
        """
        try:
            if self.model is None:
                logger.warning("TransNetV2模型未加载，使用基本的场景检测方法")
                return self._basic_scene_detection(video_path, threshold)

            # 提取帧
            frames = self.extract_video_frames(video_path)

            if self.model_type == 'pytorch':
                return self._predict_pytorch(frames, threshold)
            else:
                return self._basic_scene_detection(video_path, threshold)

        except Exception as e:
            logger.error(f"预测场景转换时发生错误: {e}")
            # 降级到基本方法
            return self._basic_scene_detection(video_path, threshold)

    def _predict_pytorch(
        self, frames: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """使用PyTorch版本进行预测"""
        try:
            import torch

            # 添加批次维度
            input_tensor = torch.tensor(frames).unsqueeze(0)  # [1, frames, H, W, 3]

            with torch.no_grad():
                if self.model is not None:
                    single_frame_pred, all_frame_pred = self.model(input_tensor)
                    single_frame_pred = (
                        torch.sigmoid(single_frame_pred).cpu().numpy()[0, :, 0]
                    )
                else:
                    raise ValueError("模型未初始化")

            # 生成场景列表
            scenes = self._predictions_to_scenes(single_frame_pred, threshold)
            return single_frame_pred, scenes

        except Exception as e:
            logger.error(f"PyTorch预测失败: {e}")
            raise

    def _predictions_to_scenes(
        self, predictions: np.ndarray, threshold: float
    ) -> List[Tuple[int, int]]:
        """将预测结果转换为场景列表"""
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        i = 0  # 初始化变量
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append((start, i))
            t_prev = t
        if t == 0:
            scenes.append((start, i))

        # 如果所有预测都是1，返回整个视频
        if len(scenes) == 0:
            return [(0, len(predictions) - 1)]

        return scenes

    def _basic_scene_detection(
        self, video_path: str, threshold: float = 0.3
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        基本的场景检测方法（基于帧差）
        当TransNetV2不可用时使用
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            frames = []
            frame_diffs = []
            prev_gray = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)

                if prev_gray is not None:
                    # 计算帧差
                    diff = cv2.absdiff(prev_gray, gray)
                    diff_array = np.array(diff)  # 确保类型为numpy数组
                    diff_mean = float(np.mean(diff_array)) / 255.0  # type: ignore  # 归一化到0-1
                    frame_diffs.append(diff_mean)
                else:
                    frame_diffs.append(0.0)

                prev_gray = gray

            cap.release()

            if not frame_diffs:
                return np.array([]), [(0, 0)]

            frame_diffs = np.array(frame_diffs)

            # 使用阈值检测场景转换
            transitions = frame_diffs > threshold

            # 生成场景列表
            scenes = []
            start = 0
            for i, is_transition in enumerate(transitions):
                if is_transition and i > 0:
                    scenes.append((start, i))
                    start = i

            # 添加最后一个场景
            if len(frames) > 0:
                scenes.append((start, len(frames) - 1))

            # 如果没有检测到场景，返回整个视频作为一个场景
            if not scenes:
                scenes = [(0, len(frames) - 1)]

            return frame_diffs, scenes

        except Exception as e:
            logger.error(f"基本场景检测失败: {e}")
            return np.array([]), [(0, 0)]

    def process_video(
        self, video_path: str, output_dir: Optional[str] = None, threshold: float = 0.5
    ) -> Dict:
        """
        处理视频并返回分析结果

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录（可选）
            threshold: 检测阈值

        Returns:
            包含分析结果的字典
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")

            # 获取视频信息
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            # 预测场景转换
            predictions, scenes = self.predict_transitions(video_path, threshold)

            # 格式化结果
            result = {
                "video_path": video_path,
                "video_info": {
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": duration,
                },
                "scene_detection": {
                    "method": self.model_type if self.model_type else "basic",
                    "threshold": threshold,
                    "total_scenes": len(scenes),
                    "scenes": [
                        {
                            "scene_id": i + 1,
                            "start_frame": scene[0],
                            "end_frame": scene[1],
                            "start_time": scene[0] / fps if fps > 0 else 0,
                            "end_time": scene[1] / fps if fps > 0 else 0,
                            "duration": (scene[1] - scene[0]) / fps if fps > 0 else 0,
                        }
                        for i, scene in enumerate(scenes)
                    ],
                },
                "predictions": predictions.tolist() if len(predictions) > 0 else [],
            }

            # 保存结果（如果指定了输出目录）
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                result_file = os.path.join(
                    output_dir, f"{Path(video_path).stem}_analysis.json"
                )
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"分析结果已保存到: {result_file}")

            return result

        except Exception as e:
            logger.error(f"处理视频时发生错误: {e}")
            raise


# 全局实例
transnetv2_service = TransNetV2Service()
