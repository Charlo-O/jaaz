"""
Video Analysis Tool using Google Gemini
Analyzes uploaded video content using Gemini's video understanding capabilities
"""

import base64
from typing import Dict, Any
from langchain_core.tools import tool
from services.gemini_video_service import get_gemini_video_service
from services.config_service import config_service


@tool("analyze_video_by_gemini")
def analyze_video_by_gemini(
    video_data: str,
    filename: str = "video.mp4",
    analysis_prompt: str = ""
) -> str:
    """
    Analyze video content using Google Gemini's video understanding capabilities.
    
    Args:
        video_data: Base64 encoded video data or data URL
        filename: Video filename for proper MIME type detection  
        analysis_prompt: Custom analysis prompt (optional)
        
    Returns:
        str: Detailed video analysis result from Gemini
    """
    
    try:
        # Get Gemini video service
        gemini_service = get_gemini_video_service()
        if not gemini_service:
            return "❌ Error: Gemini Video Service not available. Please configure Google AI API key in settings."
        
        # Extract video bytes from base64 data
        if video_data.startswith('data:'):
            # Handle data URL format: data:video/mp4;base64,<data>
            try:
                header, data = video_data.split(',', 1)
                video_bytes = base64.b64decode(data)
            except Exception as e:
                return f"❌ Error: Invalid video data format - {str(e)}"
        else:
            # Handle direct base64 data
            try:
                video_bytes = base64.b64decode(video_data)
            except Exception as e:
                return f"❌ Error: Invalid base64 video data - {str(e)}"
        
        # Use default analysis prompt if none provided
        if not analysis_prompt:
            analysis_prompt = """请详细分析这个视频，提供以下信息：

1. **视频内容概述**
   - 主要内容和主题
   - 场景描述和变化
   - 时间线和关键时刻

2. **视觉元素分析**
   - 人物、物体、动作分析
   - 色彩搭配和视觉风格
   - 镜头运用和构图

3. **技术特征**
   - 视频质量和清晰度
   - 音频质量（如果有）
   - 剪辑技巧和特效

4. **情感和氛围**
   - 整体情绪和氛围
   - 表达的情感或信息
   - 观感体验

请用中文详细回答，格式清晰易读。"""
        
        # Call Gemini video analysis service
        import asyncio
        result = asyncio.run(gemini_service.analyze_video_content(
            video_content=video_bytes,
            filename=filename, 
            prompt=analysis_prompt
        ))
        
        if result.get('success'):
            return f"✅ **Gemini视频分析结果**\n\n{result.get('result', '分析完成但无内容返回')}"
        else:
            error = result.get('error', '未知错误')
            return f"❌ **视频分析失败**\n\n错误信息: {error}"
            
    except Exception as e:
        return f"❌ **视频分析工具执行失败**\n\n错误: {str(e)}"


# Tool metadata for LangChain integration
TOOL_INFO = {
    "id": "analyze_video_by_gemini",
    "display_name": "Video Analysis by Gemini",
    "type": "analysis", 
    "provider": "google",
    "tool_function": analyze_video_by_gemini,
    "description": "Analyze video content using Google Gemini's advanced video understanding capabilities",
    "supports_video": True,
    "auto_trigger": "video_upload"  # Automatically trigger when video is uploaded
}