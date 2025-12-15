"""
TransNetV2 视频场景分割服务
用于检测视频中的镜头切换点并提取关键帧
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from typing import List, Tuple, Optional
from pathlib import Path

# 添加 TransNetV2 到 Python 路径
TRANSNET_PATH = Path(__file__).parent.parent.parent / "TransNetV2" / "inference-pytorch"
sys.path.insert(0, str(TRANSNET_PATH))

from transnetv2_pytorch import TransNetV2

from services.config_service import FILES_DIR
from tools.utils.image_canvas_utils import generate_file_id


class TransNetService:
    """TransNetV2 视频场景分割服务"""
    
    def __init__(self):
        self.model: Optional[TransNetV2] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保模型已加载"""
        if self._initialized:
            return
        
        print(f"🎬 Loading TransNetV2 model on {self.device}...")
        self.model = TransNetV2()
        
        # 加载权重
        weights_path = TRANSNET_PATH / "transnetv2-pytorch-weights.pth"
        if not weights_path.exists():
            # 尝试从文件夹加载
            weights_dir = TRANSNET_PATH / "transnetv2-pytorch-weights"
            if weights_dir.exists():
                # 需要合并权重文件
                weights_path = self._merge_weights(weights_dir)
            else:
                raise FileNotFoundError(
                    f"TransNetV2 weights not found at {weights_path}. "
                    "Please run convert_weights.py first."
                )
        
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        self._initialized = True
        print("✅ TransNetV2 model loaded successfully")
    
    def _merge_weights(self, weights_dir: Path) -> Path:
        """合并分片的权重文件"""
        import glob
        
        output_path = TRANSNET_PATH / "transnetv2-pytorch-weights.pth"
        
        # 检查是否有分片文件
        parts = sorted(glob.glob(str(weights_dir / "*.pth.*")))
        if parts:
            print(f"🔧 Merging {len(parts)} weight file parts...")
            with open(output_path, 'wb') as outfile:
                for part in parts:
                    with open(part, 'rb') as infile:
                        outfile.write(infile.read())
            print("✅ Weights merged successfully")
        else:
            # 可能直接有 .pth 文件
            pth_files = list(weights_dir.glob("*.pth"))
            if pth_files:
                return pth_files[0]
            raise FileNotFoundError(f"No weight files found in {weights_dir}")
        
        return output_path
    
    def _extract_frames(self, video_path: str, target_size: Tuple[int, int] = (48, 27)) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            target_size: 目标尺寸 (width, height)，TransNetV2 需要 48x27
            
        Returns:
            (用于模型的帧数组, 原始帧列表, fps)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_for_model = []
        original_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存原始帧（BGR -> RGB）
            original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 为模型调整大小
            resized = cv2.resize(frame, target_size)
            # BGR -> RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames_for_model.append(resized)
        
        cap.release()
        
        # 转换为 numpy 数组
        frames_array = np.array(frames_for_model, dtype=np.uint8)
        
        return frames_array, original_frames, fps
    
    def detect_scenes(self, video_path: str, threshold: float = 0.5) -> List[int]:
        """
        检测视频中的场景切换点
        
        Args:
            video_path: 视频文件路径
            threshold: 场景切换检测阈值
            
        Returns:
            场景切换帧的索引列表
        """
        self._ensure_initialized()
        
        frames_array, _, fps = self._extract_frames(video_path)
        
        if len(frames_array) == 0:
            return []
        
        print(f"🎬 Processing {len(frames_array)} frames...")
        
        # 分批处理（避免内存溢出）
        batch_size = 100
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(frames_array), batch_size - 10):  # 重叠10帧
                end_idx = min(i + batch_size, len(frames_array))
                batch = frames_array[i:end_idx]
                
                # 添加 batch 维度
                batch_tensor = torch.from_numpy(batch).unsqueeze(0)
                
                if self.device == "cuda":
                    batch_tensor = batch_tensor.cuda()
                
                # 模型推理
                single_frame_pred, _ = self.model(batch_tensor)
                predictions = torch.sigmoid(single_frame_pred).cpu().numpy()[0, :, 0]
                
                if i == 0:
                    all_predictions.extend(predictions.tolist())
                else:
                    # 跳过重叠部分
                    all_predictions.extend(predictions[10:].tolist())
        
        # 找到场景切换点
        scene_changes = []
        for i, pred in enumerate(all_predictions):
            if pred > threshold:
                scene_changes.append(i)
        
        print(f"✅ Detected {len(scene_changes)} scene changes")
        return scene_changes
    
    def extract_keyframes(
        self, 
        video_path: str, 
        threshold: float = 0.5,
        min_scene_length: int = 10
    ) -> List[dict]:
        """
        提取视频关键帧（每个场景的第一帧）
        
        Args:
            video_path: 视频文件路径
            threshold: 场景切换检测阈值
            min_scene_length: 最小场景长度（帧数）
            
        Returns:
            关键帧信息列表，每个元素包含 {file_id, url, width, height, frame_index, timestamp}
        """
        self._ensure_initialized()
        
        frames_array, original_frames, fps = self._extract_frames(video_path)
        
        if len(frames_array) == 0:
            return []
        
        print(f"🎬 Processing {len(frames_array)} frames for keyframe extraction...")
        
        # 检测场景切换点
        scene_changes = self.detect_scenes(video_path, threshold)
        
        # 添加第一帧作为第一个场景的开始
        keyframe_indices = [0]
        
        # 过滤太短的场景
        for i, change_idx in enumerate(scene_changes):
            if change_idx - keyframe_indices[-1] >= min_scene_length:
                keyframe_indices.append(change_idx)
        
        print(f"📸 Extracting {len(keyframe_indices)} keyframes...")
        
        # 保存关键帧
        keyframes = []
        for frame_idx in keyframe_indices:
            if frame_idx >= len(original_frames):
                continue
            
            frame = original_frames[frame_idx]
            height, width = frame.shape[:2]
            
            # 生成文件 ID 并保存
            file_id = generate_file_id()
            filename = f"{file_id}.jpg"
            file_path = os.path.join(FILES_DIR, filename)
            
            # 保存为 JPEG
            img = Image.fromarray(frame)
            img.save(file_path, "JPEG", quality=95)
            
            # 计算时间戳
            timestamp = frame_idx / fps if fps > 0 else 0
            
            keyframes.append({
                "file_id": filename,
                "url": f"/api/file/{filename}",
                "width": width,
                "height": height,
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 2),
            })
        
        print(f"✅ Extracted {len(keyframes)} keyframes")
        return keyframes
    
    def extract_keyframes_simple(
        self, 
        video_path: str, 
        num_frames: int = 10
    ) -> List[dict]:
        """
        简单的关键帧提取（均匀采样），不使用 TransNetV2
        用于快速提取或作为备选方案
        
        Args:
            video_path: 视频文件路径
            num_frames: 要提取的帧数
            
        Returns:
            关键帧信息列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return []
        
        # 计算采样间隔
        interval = max(1, total_frames // num_frames)
        frame_indices = list(range(0, total_frames, interval))[:num_frames]
        
        keyframes = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            
            # 生成文件 ID 并保存
            file_id = generate_file_id()
            filename = f"{file_id}.jpg"
            file_path = os.path.join(FILES_DIR, filename)
            
            img = Image.fromarray(frame)
            img.save(file_path, "JPEG", quality=95)
            
            timestamp = frame_idx / fps if fps > 0 else 0
            
            keyframes.append({
                "file_id": filename,
                "url": f"/api/file/{filename}",
                "width": width,
                "height": height,
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 2),
            })
        
        cap.release()
        print(f"✅ Extracted {len(keyframes)} keyframes (simple mode)")
        return keyframes


# 全局单例
transnet_service = TransNetService()
