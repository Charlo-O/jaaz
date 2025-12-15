"""
视频上传和分析 API 路由
"""

import os
import tempfile
import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from services.config_service import FILES_DIR
from tools.utils.image_canvas_utils import generate_file_id

router = APIRouter(prefix="/api")

# 确保文件目录存在
os.makedirs(FILES_DIR, exist_ok=True)


class VideoAnalysisRequest(BaseModel):
    video_path: str
    mode: str = "transnet"  # "transnet" 或 "simple"
    threshold: float = 0.5
    num_frames: int = 10  # 仅用于 simple 模式
    min_scene_length: int = 10


@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件
    
    Returns:
        video_id: 视频文件 ID
        url: 视频文件 URL
    """
    print(f'🎬 upload_video file: {file.filename}')
    
    # 验证文件类型
    allowed_extensions = {'mp4', 'webm', 'mov', 'avi', 'mkv', 'flv', 'm4v'}
    filename = file.filename or 'video.mp4'
    extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'mp4'
    
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported video format: {extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # 生成文件 ID
    file_id = generate_file_id()
    video_filename = f"{file_id}.{extension}"
    file_path = os.path.join(FILES_DIR, video_filename)
    
    # 保存视频文件
    try:
        content = await file.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")
    
    print(f'🎬 Video saved: {file_path}')
    
    return {
        "video_id": video_filename,
        "url": f"/api/file/{video_filename}",
        "filename": filename,
    }


@router.post("/video/analyze")
async def analyze_video(
    video_id: str = Form(...),
    mode: str = Form("transnet"),
    threshold: float = Form(0.5),
    num_frames: int = Form(10),
    min_scene_length: int = Form(10),
):
    """
    分析视频并提取关键帧
    
    Args:
        video_id: 视频文件 ID（从 upload_video 返回）
        mode: 分析模式 - "transnet"（使用 TransNetV2）或 "simple"（均匀采样）
        threshold: 场景切换检测阈值（仅 transnet 模式）
        num_frames: 要提取的帧数（仅 simple 模式）
        min_scene_length: 最小场景长度（仅 transnet 模式）
        
    Returns:
        keyframes: 关键帧列表，每个包含 file_id, url, width, height, frame_index, timestamp
    """
    print(f'🎬 analyze_video: video_id={video_id}, mode={mode}')
    
    # 构建视频路径
    video_path = os.path.join(FILES_DIR, video_id)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    
    try:
        from services.transnet_service import transnet_service
        
        if mode == "transnet":
            keyframes = transnet_service.extract_keyframes(
                video_path=video_path,
                threshold=threshold,
                min_scene_length=min_scene_length,
            )
        else:
            keyframes = transnet_service.extract_keyframes_simple(
                video_path=video_path,
                num_frames=num_frames,
            )
        
        return {
            "success": True,
            "keyframes": keyframes,
            "total": len(keyframes),
            "mode": mode,
        }
        
    except FileNotFoundError as e:
        # TransNetV2 权重文件未找到，回退到简单模式
        print(f"⚠️ TransNetV2 not available, falling back to simple mode: {e}")
        from services.transnet_service import transnet_service
        
        keyframes = transnet_service.extract_keyframes_simple(
            video_path=video_path,
            num_frames=num_frames,
        )
        
        return {
            "success": True,
            "keyframes": keyframes,
            "total": len(keyframes),
            "mode": "simple",
            "warning": "TransNetV2 not available, used simple extraction",
        }
        
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@router.post("/video/extract_keyframes")
async def extract_keyframes_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("simple"),
    threshold: float = Form(0.5),
    num_frames: int = Form(10),
    min_scene_length: int = Form(10),
):
    """
    一步完成：上传视频并提取关键帧
    
    这是一个便捷接口，合并了 upload_video 和 analyze_video
    """
    # 先上传视频
    upload_result = await upload_video(file)
    video_id = upload_result["video_id"]
    
    # 然后分析
    analysis_result = await analyze_video(
        video_id=video_id,
        mode=mode,
        threshold=threshold,
        num_frames=num_frames,
        min_scene_length=min_scene_length,
    )
    
    return {
        **analysis_result,
        "video_id": video_id,
        "video_url": upload_result["url"],
    }
