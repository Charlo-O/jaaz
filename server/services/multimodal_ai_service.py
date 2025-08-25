"""
Multimodal AI Service for processing images and videos
Supports various AI models that can handle video content
"""

import asyncio
import base64
import json
from typing import Dict, Any, List, Optional, Union
import httpx
from utils.http_client import HttpClient
from services.config_service import config_service
from services.gemini_video_service import get_gemini_video_service


class MultimodalAIService:
    """Service for processing images and videos with AI models"""
    
    def __init__(self):
        """Initialize multimodal AI service"""
        self.config = config_service.app_config
        self.supported_models = {
            # OpenAI GPT-4V supports images, GPT-4o supports videos  
            'openai': {
                'image_models': ['gpt-4-vision-preview', 'gpt-4o', 'gpt-4o-mini'],
                'video_models': ['gpt-4o', 'gpt-4o-mini'],  # GPT-4o supports video
                'base_url': 'https://api.openai.com/v1',
                'endpoint': '/chat/completions'
            },
            # Claude doesn't support video yet, only images
            'anthropic': {
                'image_models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
                'video_models': [],
                'base_url': 'https://api.anthropic.com',
                'endpoint': '/v1/messages'
            },
            # Google Gemini supports both images and videos
            'google': {
                'image_models': ['gemini-pro-vision', 'gemini-1.5-pro', 'gemini-1.5-flash'],
                'video_models': ['gemini-1.5-pro', 'gemini-1.5-flash'],
                'base_url': 'https://generativelanguage.googleapis.com',
                'endpoint': '/v1beta/models/{model}:generateContent'
            }
        }
    
    def _get_model_config(self, provider: str, model: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model"""
        provider_config = self.config.get(provider, {})
        if not provider_config:
            return None
            
        return {
            'api_key': provider_config.get('api_key', ''),
            'url': provider_config.get('url', self.supported_models[provider]['base_url']),
            'model': model
        }
    
    def _supports_video(self, provider: str, model: str) -> bool:
        """Check if a model supports video processing"""
        if provider not in self.supported_models:
            return False
        return model in self.supported_models[provider]['video_models']
    
    def _supports_image(self, provider: str, model: str) -> bool:
        """Check if a model supports image processing"""
        if provider not in self.supported_models:
            return False
        return model in self.supported_models[provider]['image_models']
    
    async def process_media_with_openai(
        self, 
        config: Dict[str, Any],
        prompt: str, 
        media_content: str, 
        media_type: str
    ) -> str:
        """Process media with OpenAI models"""
        headers = {
            'Authorization': f'Bearer {config["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        # Prepare message content
        content = [
            {"type": "text", "text": prompt}
        ]
        
        if media_type == "image":
            content.append({
                "type": "image_url",
                "image_url": {"url": media_content}
            })
        elif media_type == "video":
            content.append({
                "type": "video", 
                "video": {"url": media_content}
            })
        
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "user", 
                    "content": content
                }
            ],
            "max_tokens": 1000
        }
        
        try:
            timeout = httpx.Timeout(60.0)
            async with HttpClient.create(timeout=timeout) as client:
                response = await client.post(
                    f"{config['url']}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    return f"Error: No response from {config['model']}"
                    
        except Exception as e:
            return f"Error processing {media_type} with OpenAI: {str(e)}"
    
    async def process_media_with_google(
        self, 
        config: Dict[str, Any],
        prompt: str, 
        media_content: str, 
        media_type: str
    ) -> str:
        """Process media with Google Gemini models"""
        # Extract base64 data from data URL
        if media_content.startswith('data:'):
            header, data = media_content.split(',', 1)
            mime_type = header.split(';')[0].split(':')[1]
        else:
            return "Error: Invalid media format for Google Gemini"
        
        # Prepare content for Gemini
        parts = [{"text": prompt}]
        
        if media_type == "image":
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": data
                }
            })
        elif media_type == "video":
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": data
                }
            })
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 1000,
                "temperature": 0.7
            }
        }
        
        try:
            timeout = httpx.Timeout(60.0)
            async with HttpClient.create(timeout=timeout) as client:
                url = f"{config['url']}/v1beta/models/{config['model']}:generateContent"
                response = await client.post(
                    url,
                    headers={'Content-Type': 'application/json'},
                    json=payload,
                    params={'key': config['api_key']}
                )
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return f"Error: No response from {config['model']}"
                    
        except Exception as e:
            return f"Error processing {media_type} with Google: {str(e)}"
    
    async def analyze_media(
        self, 
        media_content: str, 
        media_type: str, 
        prompt: str = "",
        preferred_provider: str = "openai",
        preferred_model: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze image or video content with AI
        
        Args:
            media_content: Base64 encoded media data URL
            media_type: "image" or "video"  
            prompt: Analysis prompt (default: generic analysis)
            preferred_provider: Preferred AI provider
            preferred_model: Preferred model name
            
        Returns:
            Dict with analysis result and metadata
        """
        
        if not prompt:
            if media_type == "image":
                prompt = "Please describe this image in detail, including objects, people, scenes, colors, and any text you can see."
            else:
                prompt = "Please describe this video in detail, including what happens, objects, people, actions, scenes, and any text you can see."
        
        # 对于视频，优先使用专门的Gemini视频服务
        if media_type == "video":
            return await self._analyze_video_with_gemini_service(media_content, prompt)
        
        # 对于图片，使用原有的逻辑
        # Find suitable model
        provider = preferred_provider
        model = preferred_model
        
        # Auto-select if not specified
        if not model:
            if provider == "openai":
                model = "gpt-4o" if media_type == "video" else "gpt-4-vision-preview"
            elif provider == "google":
                model = "gemini-1.5-pro"
            else:
                # Default fallback
                provider = "openai"
                model = "gpt-4o" if media_type == "video" else "gpt-4-vision-preview"
        
        # Check model capability
        if media_type == "video" and not self._supports_video(provider, model):
            # Fallback to video-capable model
            if provider == "openai":
                model = "gpt-4o"
            elif provider == "google":
                model = "gemini-1.5-pro"  
            else:
                return {
                    "success": False,
                    "error": f"No video-capable models available for provider: {provider}"
                }
        
        # Get model configuration
        config = self._get_model_config(provider, model)
        if not config or not config.get('api_key'):
            return {
                "success": False,
                "error": f"Model {provider}/{model} not configured"
            }
        
        # Process media
        try:
            if provider == "openai":
                result = await self.process_media_with_openai(config, prompt, media_content, media_type)
            elif provider == "google":
                result = await self.process_media_with_google(config, prompt, media_content, media_type)
            else:
                result = f"Provider {provider} not implemented yet"
            
            return {
                "success": True,
                "result": result,
                "provider": provider,
                "model": model,
                "media_type": media_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing {media_type}: {str(e)}",
                "provider": provider,
                "model": model
            }
    
    async def _analyze_video_with_gemini_service(self, media_content: str, prompt: str) -> Dict[str, Any]:
        """使用专门的Gemini视频服务分析视频"""
        try:
            gemini_service = get_gemini_video_service()
            if not gemini_service:
                return {
                    "success": False,
                    "error": "Gemini Video Service not available. Please configure Google AI API key."
                }
            
            # 从data URL中提取视频内容
            if media_content.startswith('data:'):
                header, data = media_content.split(',', 1)
                video_bytes = base64.b64decode(data)
                
                # 从header中提取文件扩展名
                mime_type = header.split(';')[0].split(':')[1]
                if mime_type == 'video/mp4':
                    filename = 'video.mp4'
                elif mime_type == 'video/webm':
                    filename = 'video.webm'
                elif mime_type == 'video/mov':
                    filename = 'video.mov'
                else:
                    filename = 'video.mp4'  # 默认
                
                # 调用Gemini视频服务
                result = await gemini_service.analyze_video_content(
                    video_content=video_bytes,
                    filename=filename,
                    prompt=prompt
                )
                
                return result
            else:
                return {
                    "success": False,
                    "error": "Invalid video data format"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error in Gemini Video Service: {str(e)}",
                "provider": "google",
                "model": "gemini-1.5-flash"
            }
    
    def get_supported_models(self) -> Dict[str, Any]:
        """Get list of supported models and their capabilities"""
        result = {}
        for provider, info in self.supported_models.items():
            config = self.config.get(provider, {})
            if config.get('api_key'):
                result[provider] = {
                    'image_models': info['image_models'],
                    'video_models': info['video_models'],
                    'configured': True
                }
            else:
                result[provider] = {
                    'image_models': info['image_models'], 
                    'video_models': info['video_models'],
                    'configured': False
                }
        return result


# Global instance
multimodal_ai_service = MultimodalAIService()