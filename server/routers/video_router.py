from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from common import DEFAULT_PORT
from tools.utils.image_canvas_utils import generate_file_id, save_image_to_canvas
from services.websocket_service import broadcast_session_update
from services.transnetv2_service import transnetv2_service
from typing import Optional
from io import BytesIO

# 在文件开头添加的导入

import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import aiofiles
from mimetypes import guess_type
import cv2
from PIL import Image
from io import BytesIO
from utils.video_compression import compress_video_with_ffmpeg, get_video_info_ffprobe
from services.config_service import FILES_DIR

router = APIRouter(prefix="/api")
os.makedirs(FILES_DIR, exist_ok=True)

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = {
    'mp4',
    'avi',
    'mov',
    'mkv',
    'wmv',
    'flv',
    'webm',
    '3gp',
    'ogv',
}


@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...), max_size_mb: float = 200.0):
    print('🎬 upload_video file', file.filename)

    # 生成文件 ID 和文件名
    file_id = generate_file_id()
    filename = file.filename or ''

    # 检查文件类型
    mime_type, _ = guess_type(filename)
    if not mime_type or not mime_type.startswith('video/'):
        # 检查扩展名
        extension = filename.split('.')[-1].lower() if '.' in filename else ''
        if extension not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}",
            )

    # Read the file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    original_size_mb = len(content) / (1024 * 1024)  # Convert to MB

    # Check file size and compress if needed
    needs_compression = original_size_mb > max_size_mb
    if needs_compression:
        print(
            f'🎬 Video size ({original_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB), attempting compression...'
        )

        try:
            # Try to compress video using FFmpeg if available
            compressed_content = await run_in_threadpool(
                compress_video_with_ffmpeg, content, max_size_mb
            )
            if compressed_content:
                content = compressed_content
                original_size_mb = len(content) / (1024 * 1024)
                print(f'🎬 Video compressed to {original_size_mb:.2f}MB')
                # Update extension for compressed video
                extension = 'mp4'  # FFmpeg output is always MP4
            else:
                print(
                    f'🎬 Video compression not available or failed, checking if within acceptable range'
                )
                # Allow slightly larger files if compression failed (up to 300MB)
                if original_size_mb > 300.0:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video file size ({original_size_mb:.2f}MB) is too large even after compression attempt. Maximum size: 300MB",
                    )
        except Exception as e:
            print(f'🎬 Video compression failed: {e}')
            # Allow original file if compression fails but is under 300MB
            if original_size_mb > 300.0:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video file size ({original_size_mb:.2f}MB) exceeds maximum size and compression failed. Maximum size: 300MB",
                )

    # Determine file extension
    if mime_type and mime_type.startswith('video/'):
        extension = mime_type.split('/')[-1]
        # Handle common video format mappings
        if extension == 'quicktime':
            extension = 'mov'
        elif extension == 'x-msvideo':
            extension = 'avi'
    else:
        extension = filename.split('.')[-1].lower() if '.' in filename else 'mp4'

    # Save video file
    file_path = os.path.join(FILES_DIR, f'{file_id}.{extension}')

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

    # Extract video metadata using FFprobe (preferred) or OpenCV (fallback)
    try:
        video_info = await run_in_threadpool(get_video_info_ffprobe, file_path)
        # If FFprobe didn't work, try OpenCV
        if video_info['duration'] == 0:
            opencv_info = await run_in_threadpool(extract_video_info, file_path)
            video_info.update(opencv_info)
    except Exception as e:
        print(f'🎬 Error extracting video info: {e}')
        video_info = {'width': 1920, 'height': 1080, 'duration': 0, 'fps': 30.0}

    # Generate video thumbnail
    try:
        thumbnail_path = await run_in_threadpool(
            generate_video_thumbnail, file_path, file_id
        )
        thumbnail_url = (
            f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}_thumb.jpg'
            if thumbnail_path
            else None
        )
    except Exception as e:
        print(f'🎬 Error generating thumbnail: {e}')
        thumbnail_url = None

    print('🎬 upload_video file_path', file_path)

    return {
        'file_id': f'{file_id}.{extension}',
        'url': f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}.{extension}',
        'thumbnail_url': thumbnail_url,
        'width': video_info['width'],
        'height': video_info['height'],
        'duration': video_info['duration'],
        'fps': video_info['fps'],
        'size_mb': original_size_mb,
        'type': 'video',
    }


def extract_video_info(file_path: str) -> dict:
    """Extract video metadata using OpenCV"""
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count / fps if fps > 0 else 0

        return {
            'width': width,
            'height': height,
            'duration': round(duration, 2),
            'fps': round(fps, 2),
        }
    finally:
        cap.release()


def generate_video_thumbnail(file_path: str, file_id: str) -> str:
    """Generate a thumbnail from the first frame of the video"""
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file for thumbnail generation")

    try:
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")

        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Resize thumbnail to a reasonable size (max 300x300, maintain aspect ratio)
        thumbnail_size = (300, 300)
        pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

        # Save thumbnail
        thumbnail_path = os.path.join(FILES_DIR, f'{file_id}_thumb.jpg')
        pil_image.save(thumbnail_path, format='JPEG', quality=85, optimize=True)

        return thumbnail_path

    finally:
        cap.release()


@router.post("/process_video")
async def process_video_direct(file: UploadFile = File(...), threshold: float = 0.5):
    """
    直接处理上传的视频文件，进行场景分割分析
    无需调用大模型，直接使用TransNetV2进行视频处理
    """
    print('🎬 process_video_direct file', file.filename)

    # 生成文件 ID 和文件名
    file_id = generate_file_id()
    filename = file.filename or ''

    # 检查文件类型
    mime_type, _ = guess_type(filename)
    if not mime_type or not mime_type.startswith('video/'):
        # 检查扩展名
        extension = filename.split('.')[-1].lower() if '.' in filename else ''
        if extension not in SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}",
            )

    # Read the file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # 确定文件扩展名
    if mime_type and mime_type.startswith('video/'):
        extension = mime_type.split('/')[-1]
        # Handle common video format mappings
        if extension == 'quicktime':
            extension = 'mov'
        elif extension == 'x-msvideo':
            extension = 'avi'
    else:
        extension = filename.split('.')[-1].lower() if '.' in filename else 'mp4'

    # 保存视频文件
    file_path = os.path.join(FILES_DIR, f'{file_id}.{extension}')

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

    try:
        # 使用TransNetV2处理视频
        analysis_result = await run_in_threadpool(
            transnetv2_service.process_video, file_path, threshold=threshold
        )

<<<<<<< Updated upstream
=======
        # 过滤分析结果中的base64数据，避免在终端显示
        filtered_analysis = transnetv2_service._filter_base64_from_data(analysis_result)
        print('🎬 视频分析结果（已过滤）:', filtered_analysis)

>>>>>>> Stashed changes
        # 生成缩略图
        thumbnail_path = await run_in_threadpool(
            generate_video_thumbnail, file_path, file_id
        )
        thumbnail_url = (
            f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}_thumb.jpg'
            if thumbnail_path
            else None
        )

        print('🎬 process_video_direct completed', file_path)

        return {
            'file_id': f'{file_id}.{extension}',
            'url': f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}.{extension}',
            'thumbnail_url': thumbnail_url,
            'type': 'video',
            'analysis': analysis_result,
            'message': f'视频处理完成，检测到 {analysis_result["scene_detection"]["total_scenes"]} 个场景',
        }

    except Exception as e:
        print(f'🎬 Error processing video: {e}')
        # 如果处理失败，至少返回基本的视频信息
        try:
            video_info = await run_in_threadpool(get_video_info_ffprobe, file_path)
            thumbnail_path = await run_in_threadpool(
                generate_video_thumbnail, file_path, file_id
            )
            thumbnail_url = (
                f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}_thumb.jpg'
                if thumbnail_path
                else None
            )

            return {
                'file_id': f'{file_id}.{extension}',
                'url': f'http://localhost:{DEFAULT_PORT}/api/file/{file_id}.{extension}',
                'thumbnail_url': thumbnail_url,
                'width': video_info['width'],
                'height': video_info['height'],
                'duration': video_info['duration'],
                'fps': video_info['fps'],
                'type': 'video',
                'message': f'视频上传成功，但场景分析失败: {str(e)}',
            }
        except:
            raise HTTPException(
                status_code=500, detail=f"Video processing failed: {str(e)}"
            )


@router.get("/video_analysis/{file_id}")
async def get_video_analysis(file_id: str):
    """
    获取已处理视频的分析结果
    """
    try:
        # 从文件系统中查找对应的视频文件
        for ext in SUPPORTED_VIDEO_FORMATS:
            video_path = os.path.join(FILES_DIR, f'{file_id}.{ext}')
            if os.path.exists(video_path):
                # 重新分析视频（或从缓存中获取）
                analysis_result = await run_in_threadpool(
                    transnetv2_service.process_video, video_path
                )
<<<<<<< Updated upstream
=======
                # 过滤分析结果中的base64数据
                filtered_result = transnetv2_service._filter_base64_from_data(
                    analysis_result
                )
                print('🎬 视频分析结果（已过滤）:', filtered_result)
>>>>>>> Stashed changes
                return analysis_result

        raise HTTPException(status_code=404, detail="Video file not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")


@router.post("/extract_scenes/{file_id}")
async def extract_video_scenes(file_id: str, scene_indices: Optional[list] = None):
    """
    从视频中提取指定的场景片段
    """
    try:
        # 查找视频文件
        video_path = None
        for ext in SUPPORTED_VIDEO_FORMATS:
            path = os.path.join(FILES_DIR, f'{file_id}.{ext}')
            if os.path.exists(path):
                video_path = path
                break

        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")

        # 获取视频分析结果
        analysis_result = await run_in_threadpool(
            transnetv2_service.process_video, video_path
        )

        scenes = analysis_result["scene_detection"]["scenes"]

        # 如果没有指定场景索引，返回所有场景信息
        if scene_indices is None:
            scene_indices = list(range(len(scenes)))

        # 提取指定的场景信息
        extracted_scenes = []
        for idx in scene_indices:
            if 0 <= idx < len(scenes):
                scene = scenes[idx]
                extracted_scenes.append(
                    {
                        "scene_index": idx,
                        "scene_info": scene,
                        "extract_url": f'http://localhost:{DEFAULT_PORT}/api/extract_scene/{file_id}/{idx}',
                    }
                )

        return {
            "video_file_id": file_id,
            "total_scenes": len(scenes),
            "extracted_scenes": extracted_scenes,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error extracting scenes: {str(e)}"
        )


@router.get("/extract_scene/{file_id}/{scene_index}")
async def extract_single_scene(file_id: str, scene_index: int):
    """
    提取单个场景并返回视频片段
    """
    try:
        # 查找视频文件
        video_path = None
        original_ext = None
        for ext in SUPPORTED_VIDEO_FORMATS:
            path = os.path.join(FILES_DIR, f'{file_id}.{ext}')
            if os.path.exists(path):
                video_path = path
                original_ext = ext
                break

        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")

        # 获取场景信息
        analysis_result = await run_in_threadpool(
            transnetv2_service.process_video, video_path
        )

        scenes = analysis_result["scene_detection"]["scenes"]

        if scene_index < 0 or scene_index >= len(scenes):
            raise HTTPException(status_code=400, detail="Invalid scene index")

        scene = scenes[scene_index]

        # 生成场景片段文件
        scene_file_id = f"{file_id}_scene_{scene_index}"
        scene_file_path = os.path.join(FILES_DIR, f'{scene_file_id}.{original_ext}')

        # 如果场景文件不存在，则创建
        if not os.path.exists(scene_file_path):
            await run_in_threadpool(
                extract_scene_segment,
                video_path,
                scene_file_path,
                scene["start_time"],
                scene["end_time"],
            )

        return FileResponse(
            scene_file_path,
            media_type=f"video/{original_ext}",
            filename=f"scene_{scene_index}_{scene_file_id}.{original_ext}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting scene: {str(e)}")


def extract_scene_segment(
    input_path: str, output_path: str, start_time: float, end_time: float
):
    """
    从视频中提取指定时间段的片段
    """
    try:
        # 使用OpenCV提取视频片段
        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定义编解码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 定位到开始帧
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()

    except Exception as e:
        print(f"Error extracting scene segment: {e}")
        raise


@router.post("/analyze_video_to_canvas")
async def analyze_video_to_canvas(request: Request):
    """
    分析视频并将结果图片添加到画布
    """
    try:
        request_data = await request.json()
        file_id = request_data.get('file_id')
        canvas_id = request_data.get('canvas_id')
        threshold = request_data.get('threshold', 0.5)

        print(
            f'🎬 analyze_video_to_canvas 接收到的参数: file_id={file_id}, canvas_id={canvas_id}, threshold={threshold}'
        )
        print(f'🎬 request_data: {request_data}')

        if not file_id or not canvas_id:
            print(f'❌ 参数检查失败: file_id={file_id}, canvas_id={canvas_id}')
            raise HTTPException(status_code=400, detail="Missing file_id or canvas_id")

        # 查找视频文件
        video_path = None

        # 首先尝试直接使用file_id查找（因为upload_video返回的file_id已包含扩展名）
        direct_path = os.path.join(FILES_DIR, file_id)
        if os.path.exists(direct_path):
            video_path = direct_path
        else:
            # 如果直接查找失败，尝试添加各种扩展名
            for ext in SUPPORTED_VIDEO_FORMATS:
                path = os.path.join(FILES_DIR, f'{file_id}.{ext}')
                if os.path.exists(path):
                    video_path = path
                    break

        if not video_path:
            raise HTTPException(
                status_code=404, detail=f"Video file not found: {file_id}"
            )

        # 使用TransNetV2分析视频
        analysis_result = await run_in_threadpool(
            transnetv2_service.process_video, video_path, threshold=threshold
        )

<<<<<<< Updated upstream
=======
        # 过滤分析结果中的base64数据
        filtered_analysis = transnetv2_service._filter_base64_from_data(analysis_result)
        print('🎬 视频分析结果（已过滤）:', filtered_analysis)

>>>>>>> Stashed changes
        # 获取场景信息
        scenes = analysis_result["scene_detection"]["scenes"]

        # 将场景信息转换为(开始帧, 结束帧)格式用于关键帧提取
        scene_ranges = [(scene['start_frame'], scene['end_frame']) for scene in scenes]

        # 提取关键帧
        print(f'🎬 开始从 {len(scenes)} 个场景中提取关键帧...')
        key_frames = await run_in_threadpool(
            transnetv2_service.extract_key_frames, video_path, scene_ranges
        )

        # 保存关键帧并添加到画布
        saved_images = []
        base_file_id = file_id.split('.')[0] if '.' in file_id else file_id

        for i, frame in enumerate(key_frames):
            # 保存关键帧为图片文件
            frame_filename = f'{base_file_id}_keyframe_{i}.png'
            frame_file_path = os.path.join(FILES_DIR, frame_filename)

            # 转换numpy数组为PIL图像并保存
            from PIL import Image

            frame_image = Image.fromarray(frame)
            frame_image.save(frame_file_path, 'PNG')

            print(f'🎨 保存关键帧 {i+1}/{len(key_frames)}: {frame_filename}')

            # 获取图片尺寸
            width, height = frame_image.size

            # 将关键帧添加到画布
            print(f'🎨 开始将关键帧添加到画布: {frame_filename}')

            image_url = await save_image_to_canvas(
                session_id=request_data.get('session_id', canvas_id),
                canvas_id=canvas_id,
                filename=frame_filename,
                mime_type='image/png',
                width=width,
                height=height,
            )

            saved_images.append(
                {
                    'filename': frame_filename,
                    'url': image_url,
                    'scene_index': i,
                    'width': width,
                    'height': height,
                }
            )

            print(f'🎨 关键帧 {i+1} 保存完成: {image_url}')

        # 创建简化的分析结果，移除可能导致前端显示问题的predictions数组
        simplified_analysis = {
            "video_path": analysis_result["video_path"],
            "video_info": analysis_result["video_info"],
            "scene_detection": {
                "method": analysis_result["scene_detection"]["method"],
                "threshold": analysis_result["scene_detection"]["threshold"],
                "total_scenes": analysis_result["scene_detection"]["total_scenes"],
                "scenes": analysis_result["scene_detection"]["scenes"],
                # 注意：移除了predictions数组，这可能导致前端显示base64编码
            },
        }

        return {
            'success': True,
            'message': f'视频分析完成，检测到 {len(scenes)} 个场景，已提取 {len(key_frames)} 个关键帧并添加到画布',
            'analysis': simplified_analysis,
            'key_frames': saved_images,
            'total_scenes': len(scenes),
            'total_key_frames': len(key_frames),
        }

    except Exception as e:
        print(f'❌ Error analyzing video to canvas: {e}')
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
