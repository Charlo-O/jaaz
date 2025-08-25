"""
Google Gemini Video Analysis Service
Based on the official Gemini File API for video processing
"""

import os
import time
import mimetypes
import requests
import tempfile
from typing import Dict, Any, Optional
from services.config_service import config_service
from utils.http_client import HttpClient
import httpx


class GeminiVideoService:
    """Google Gemini专用视频分析服务"""
    
    def __init__(self):
        """初始化Gemini视频服务"""
        config = config_service.app_config.get('google', {})
        self.api_key = config.get('api_key', '')
        
        if not self.api_key:
            raise ValueError("Google AI API key not configured")
        
        self.base_url = "https://generativelanguage.googleapis.com"
        self.upload_url = f"{self.base_url}/upload/v1beta/files"
        self.generate_url = f"{self.base_url}/v1beta/models/gemini-1.5-flash:generateContent"
        
        print(f"✅ Gemini Video Service initialized")
    
    async def upload_video_file(self, video_content: bytes, filename: str) -> str:
        """
        上传视频文件到Gemini File API
        
        Args:
            video_content: 视频文件字节内容
            filename: 文件名
            
        Returns:
            str: 上传文件的URI
        """
        # 获取MIME类型
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith('video/'):
            mime_type = 'video/mp4'  # 默认类型
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(video_content)
            temp_file.flush()
            
            return await self._upload_file_resumable(temp_file.name, filename, mime_type)
    
    async def _upload_file_resumable(self, file_path: str, display_name: str, mime_type: str) -> str:
        """
        使用resumable upload上传文件到Gemini
        
        Args:
            file_path: 本地文件路径
            display_name: 显示名称
            mime_type: MIME类型
            
        Returns:
            str: 上传文件的name字段
        """
        file_size = os.path.getsize(file_path)
        
        # Step 1: 初始化上传
        headers = {
            'X-Goog-Upload-Protocol': 'resumable',
            'X-Goog-Upload-Command': 'start',
            'X-Goog-Upload-Header-Content-Length': str(file_size),
            'X-Goog-Upload-Header-Content-Type': mime_type,
            'Content-Type': 'application/json'
        }
        
        initial_metadata = {
            'file': {
                'display_name': display_name
            }
        }
        
        timeout = httpx.Timeout(30.0)
        async with HttpClient.create(timeout=timeout) as client:
            response = await client.post(
                f"{self.upload_url}?key={self.api_key}",
                headers=headers,
                json=initial_metadata
            )
            
            if response.status_code != 200:
                raise Exception(f"初始化上传失败: {response.status_code} - {response.text}")
            
            upload_url = response.headers.get('X-Goog-Upload-URL')
            if not upload_url:
                raise Exception("无法获取上传URL")
            
            # Step 2: 上传文件内容
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            upload_headers = {
                'Content-Length': str(len(file_data)),
                'X-Goog-Upload-Offset': '0',
                'X-Goog-Upload-Command': 'upload, finalize'
            }
            
            upload_response = await client.post(
                upload_url,
                headers=upload_headers,
                content=file_data
            )
            
            if upload_response.status_code != 200:
                raise Exception(f"上传文件失败: {upload_response.status_code} - {upload_response.text}")
            
            result = upload_response.json()
            return result['file']['name']
    
    async def wait_for_file_processing(self, file_name: str, max_wait_time: int = 300) -> bool:
        """
        等待文件处理完成
        
        Args:
            file_name: 文件名
            max_wait_time: 最大等待时间（秒）
            
        Returns:
            bool: 是否处理成功
        """
        start_time = time.time()
        
        timeout = httpx.Timeout(10.0)
        async with HttpClient.create(timeout=timeout) as client:
            while True:
                if time.time() - start_time > max_wait_time:
                    raise Exception(f"文件处理超时: {max_wait_time}秒")
                
                status_url = f"{self.base_url}/v1beta/{file_name}?key={self.api_key}"
                response = await client.get(status_url)
                
                if response.status_code != 200:
                    raise Exception(f"检查文件状态失败: {response.status_code} - {response.text}")
                
                file_info = response.json()
                state = file_info.get('state', 'UNKNOWN')
                
                print(f"🔄 File processing state: {state}")
                
                if state == 'ACTIVE':
                    return True
                elif state == 'FAILED':
                    error = file_info.get('error', {})
                    raise Exception(f"文件处理失败: {error}")
                elif state in ['PROCESSING', 'UNKNOWN']:
                    await asyncio.sleep(2)  # 等待2秒后重试
                else:
                    await asyncio.sleep(1)  # 其他状态等待1秒
    
    async def analyze_video_with_prompt(self, file_name: str, prompt: str) -> str:
        """
        使用指定提示词分析视频
        
        Args:
            file_name: 已上传的文件名
            prompt: 分析提示词
            
        Returns:
            str: 分析结果
        """
        request_data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "fileData": {
                            "mimeType": "video/mp4",
                            "fileUri": f"https://generativelanguage.googleapis.com/v1beta/{file_name}"
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 8000
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        timeout = httpx.Timeout(60.0)
        async with HttpClient.create(timeout=timeout) as client:
            response = await client.post(
                f"{self.generate_url}?key={self.api_key}",
                json=request_data,
                headers=headers
            )
            
            if response.status_code != 200:
                try:
                    error_detail = response.json()
                    error_message = error_detail.get('error', {}).get('message', response.text)
                    raise Exception(f"Gemini API调用失败: {response.status_code} - {error_message}")
                except:
                    raise Exception(f"Gemini API调用失败: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if 'candidates' not in result or len(result['candidates']) == 0:
                raise Exception("Gemini API返回无效响应")
            
            return result['candidates'][0]['content']['parts'][0]['text']
    
    async def delete_uploaded_file(self, file_name: str):
        """删除已上传的文件"""
        try:
            delete_url = f"{self.base_url}/v1beta/{file_name}?key={self.api_key}"
            
            timeout = httpx.Timeout(10.0)
            async with HttpClient.create(timeout=timeout) as client:
                await client.delete(delete_url)
            
            print(f"🗑️ Deleted uploaded file: {file_name}")
        except Exception as e:
            print(f"⚠️ Failed to delete uploaded file: {e}")
    
    async def analyze_video_content(
        self, 
        video_content: bytes, 
        filename: str,
        prompt: str = None
    ) -> Dict[str, Any]:
        """
        完整的视频分析流程
        
        Args:
            video_content: 视频文件字节内容
            filename: 文件名
            prompt: 自定义分析提示词
            
        Returns:
            Dict: 分析结果
        """
        if not prompt:
            prompt = """请详细分析这个视频，提供以下信息：

1. **视频内容分析**
   - 主要内容和主题
   - 场景描述和变化
   - 人物、物体、动作分析
   - 情感和氛围

2. **视觉效果分析**
   - 色彩搭配和调色风格
   - 光影效果和拍摄角度
   - 构图和镜头运用

3. **技术分析**
   - 视频质量和清晰度
   - 剪辑技巧
   - 音频质量（如果有）

4. **创意和风格**
   - 整体风格定位
   - 创意亮点

请用中文回答，格式要清晰易读。"""
        
        file_name = None
        try:
            # 1. 上传视频文件
            print("🔄 Uploading video to Gemini...")
            file_name = await self.upload_video_file(video_content, filename)
            print(f"✅ Video uploaded: {file_name}")
            
            # 2. 等待文件处理完成
            print("🔄 Waiting for file processing...")
            await self.wait_for_file_processing(file_name)
            print("✅ File processing completed")
            
            # 3. 分析视频
            print("🔄 Analyzing video content...")
            analysis_result = await self.analyze_video_with_prompt(file_name, prompt)
            print("✅ Video analysis completed")
            
            return {
                "success": True,
                "result": analysis_result,
                "provider": "google",
                "model": "gemini-1.5-flash",
                "media_type": "video"
            }
            
        except Exception as e:
            print(f"❌ Video analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "google",
                "model": "gemini-1.5-flash",
                "media_type": "video"
            }
        
        finally:
            # 4. 清理上传的文件
            if file_name:
                try:
                    await self.delete_uploaded_file(file_name)
                except Exception as e:
                    print(f"⚠️ Failed to cleanup file: {e}")


# 需要导入asyncio
import asyncio

# 全局实例
_gemini_video_service = None

def get_gemini_video_service() -> Optional[GeminiVideoService]:
    """获取Gemini视频服务实例"""
    global _gemini_video_service
    if _gemini_video_service is None:
        try:
            _gemini_video_service = GeminiVideoService()
        except ValueError as e:
            print(f"⚠️ Gemini Video Service not available: {e}")
            return None
    return _gemini_video_service