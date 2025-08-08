import os
import json
import traceback
from typing import Optional, Any, Dict
from .image_base_provider import ImageProviderBase
from ..utils.image_utils import get_image_info_and_save, generate_image_id
from services.config_service import FILES_DIR, config_service
from utils.http_client import HttpClient


class ModelScopeImageProvider(ImageProviderBase):
    """ModelScope API image generation provider implementation"""

    def _get_api_config(self) -> Dict[str, str]:
        """Get ModelScope API configuration"""
        config = config_service.app_config.get('modelscope', {})
        
        api_key = str(config.get("api_key", ""))
        # 使用官方文档中的标准API端点
        base_url = str(config.get("base_url", "https://api-inference.modelscope.cn/v1/images/generations"))
        
        if not api_key:
            raise ValueError("ModelScope API key is not configured")
            
        return {
            "api_key": api_key,
            "base_url": base_url
        }

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        """Build request headers"""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _convert_aspect_ratio_to_size(self, aspect_ratio: str) -> str:
        """Convert aspect ratio to ModelScope size format"""
        aspect_ratio_map = {
            "1:1": "1024x1024",
            "16:9": "1920x1080", 
            "9:16": "1080x1920",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "3:2": "1024x684",
            "2:3": "684x1024"
        }
        return aspect_ratio_map.get(aspect_ratio, "1024x1024")

    async def _make_request(self, url: str, headers: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request and handle response"""
        async with HttpClient.create_aiohttp() as session:
            print(f'🔥 ModelScope API request: {url}, model: {data["model"]}, prompt: {data["prompt"]}')
            
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            
            async with session.post(url, headers=headers, data=json_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"HTTP {response.status}: {error_text}"
                    print(f'🔥 ModelScope API error: {error_msg}')
                    raise Exception(f'Image generation failed: {error_msg}')

                json_response = await response.json()
                print('🔥 ModelScope API response:', json_response)
                
                return json_response

    async def _process_response(
        self, 
        response: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, int, int, str]:
        """Process ModelScope response and save image"""
        try:
            # ModelScope returns images array with url field
            images = response.get('images', [])
            if not images:
                raise Exception('No images found in ModelScope response')
            
            # Get first image URL
            image_url = images[0].get('url')
            if not image_url:
                raise Exception('No image URL found in ModelScope response')

            print(f'🔥 ModelScope image URL: {image_url}')
            
            # Download and save the image
            image_id = generate_image_id()
            mime_type, width, height, extension = await get_image_info_and_save(
                image_url,
                os.path.join(FILES_DIR, f'{image_id}'),
                metadata=metadata
            )

            filename = f'{image_id}.{extension}'
            return mime_type, width, height, filename

        except Exception as e:
            print(f'🔥 Error processing ModelScope response: {e}')
            raise Exception(f'ModelScope image generation failed: {e}')

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> tuple[str, int, int, str]:
        """
        Generate image using ModelScope API
        
        Args:
            prompt: Image generation prompt
            model: ModelScope model ID (e.g., 'MAILAND/majicflus_v1')
            aspect_ratio: Image aspect ratio
            input_images: Optional input images (not supported by ModelScope text-to-image)
            metadata: Optional metadata to be saved in PNG info
            **kwargs: Additional parameters like negative_prompt, steps, guidance, seed
            
        Returns:
            tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        try:
            # Get API configuration
            api_config = self._get_api_config()
            headers = self._build_headers(api_config["api_key"])
            
            # Build request data based on ModelScope API format
            data = {
                "model": model,
                "prompt": prompt,
                "size": self._convert_aspect_ratio_to_size(aspect_ratio)
            }
            
            # Add optional parameters if provided
            if kwargs.get("negative_prompt"):
                data["negative_prompt"] = kwargs["negative_prompt"]
            if kwargs.get("seed") is not None:
                data["seed"] = int(kwargs["seed"])
            if kwargs.get("steps") is not None:
                data["steps"] = int(kwargs["steps"])
            if kwargs.get("guidance") is not None:
                data["guidance"] = float(kwargs["guidance"])
                
            # Note: ModelScope text-to-image doesn't support input_images
            if input_images:
                print("⚠️ Warning: ModelScope text-to-image API doesn't support input images, ignoring...")

            # Make API request
            response = await self._make_request(api_config["base_url"], headers, data)
            
            # Process response and save image
            return await self._process_response(response, metadata)

        except Exception as e:
            print(f'Error generating image with ModelScope: {e}')
            traceback.print_exc()
            raise e