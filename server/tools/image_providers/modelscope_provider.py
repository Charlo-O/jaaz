import os
import traceback
import json
from typing import Optional, Any, Dict
from .image_base_provider import ImageProviderBase
from ..utils.image_utils import get_image_info_and_save, generate_image_id
from services.config_service import FILES_DIR
from utils.http_client import HttpClient
from services.config_service import config_service


class ModelScopeImageProvider(ImageProviderBase):
    """ModelScope (魔搭) image generation provider implementation"""

    def _get_api_config(self) -> Dict[str, str]:
        """Get API configuration from config service"""
        config = config_service.app_config.get('modelscope', {})
        api_key = str(config.get("api_key", ""))
        base_url = str(
            config.get("url", "https://api-inference.modelscope.cn/v1")
        ).rstrip("/")

        if not api_key:
            raise ValueError(
                "ModelScope API key is not configured. Please get your access token from https://modelscope.cn/my/myaccesstoken"
            )

        return {"api_key": api_key, "base_url": base_url}

    def _build_url(self) -> str:
        """Build request URL for image generation"""
        config = self._get_api_config()
        return f"{config['base_url']}/images/generations"

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers"""
        config = self._get_api_config()

        return {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "User-Agent": "Jaaz/1.0",
        }

    def _map_aspect_ratio_to_size(self, aspect_ratio: str) -> str:
        """Map aspect ratio to ModelScope size format"""
        size_map = {
            "1:1": "1024x1024",
            "16:9": "1792x1008",
            "9:16": "1008x1792",
            "4:3": "1024x768",
            "3:4": "768x1024",
        }
        return size_map.get(aspect_ratio, "1024x1024")

    async def _make_request(
        self, url: str, headers: Dict[str, str], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send HTTP request and handle response"""
        # Use longer timeout for image generation (10 minutes)
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=600, connect=60)
        async with HttpClient.create_aiohttp(timeout=timeout) as session:
            print(f'🎨 ModelScope API request: {url}')
            print(
                f'🎨 Model: {data.get("model", "")}, Prompt: {data.get("prompt", "")}'
            )

            # Convert data to JSON string with UTF-8 encoding
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')

            async with session.post(url, headers=headers, data=json_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"HTTP {response.status}: {error_text}"
                    print(f'🎨 ModelScope API error: {error_msg}')

                    # Provide more user-friendly error messages
                    if response.status == 504:
                        user_msg = "ModelScope 服务器繁忙或响应超时，请稍后重试。这是服务端的问题，不是程序问题。"
                    elif response.status == 503:
                        user_msg = "ModelScope 服务暂时不可用，请稍后重试。"
                    elif response.status == 429:
                        user_msg = "请求过于频繁，请稍后重试。"
                    elif response.status == 400:
                        user_msg = "请求参数有误或模型不存在，请检查模型名称是否正确。"
                    elif response.status == 401:
                        user_msg = "API 密钥无效，请检查 ModelScope API 配置。"
                    else:
                        user_msg = (
                            f"ModelScope API 错误 ({response.status})，请稍后重试。"
                        )

                    raise Exception(user_msg)

                response_data = await response.json()
                print('🎨 ModelScope API response received successfully')
                return response_data

    async def _process_response(
        self, response_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, int, int, str]:
        """
        Process ModelScope API response and save image

        According to ModelScope API documentation, the response format is:
        {
            "images": [
                {
                    "url": "https://..."
                }
            ]
        }

        Args:
            response_data: API response data
            metadata: Optional metadata to embed in image

        Returns:
            tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        # Check for images array in response
        if 'images' not in response_data or not isinstance(
            response_data['images'], list
        ):
            raise Exception(
                "Invalid response format: 'images' array not found in ModelScope API response"
            )

        if len(response_data['images']) == 0:
            raise Exception("No images returned from ModelScope API")

        # Get the first image
        image_data = response_data['images'][0]

        if 'url' not in image_data:
            raise Exception("No 'url' field found in image data from ModelScope API")

        image_url = image_data['url']
        print(f'🎨 Image URL received: {image_url}')

        # Generate unique image ID and save
        image_id = generate_image_id()

        # Download and save the image
        mime_type, width, height, extension = await get_image_info_and_save(
            image_url, os.path.join(FILES_DIR, f'{image_id}'), metadata=metadata
        )

        if mime_type is None:
            raise Exception(
                'Failed to determine image MIME type from ModelScope response'
            )

        filename = f'{image_id}.{extension}'
        print(f'🎨 Image saved as: {filename} ({width}x{height})')

        return mime_type, width, height, filename

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[str, int, int, str]:
        """
        Generate image using ModelScope API

        Args:
            prompt: Image generation prompt
            model: ModelScope model ID (e.g., 'MAILAND/majicflus_v1')
            aspect_ratio: Image aspect ratio
            input_images: Not supported by ModelScope text-to-image API
            metadata: Optional metadata to embed
            **kwargs: Additional parameters (negative_prompt, steps, guidance, seed)

        Returns:
            tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        try:
            # ModelScope text-to-image API doesn't support input images
            if input_images and len(input_images) > 0:
                print(
                    "⚠️ Warning: ModelScope text-to-image API doesn't support input images. Ignoring input_images parameter."
                )

            url = self._build_url()
            headers = self._build_headers()

            # Build request payload according to ModelScope API specification
            payload = {
                "model": model,
                "prompt": prompt,
                "size": self._map_aspect_ratio_to_size(aspect_ratio),
            }

            # Add optional parameters if provided
            if "negative_prompt" in kwargs and kwargs["negative_prompt"]:
                payload["negative_prompt"] = kwargs["negative_prompt"]

            if "steps" in kwargs and kwargs["steps"]:
                payload["steps"] = int(kwargs["steps"])

            if "guidance" in kwargs and kwargs["guidance"]:
                payload["guidance"] = float(kwargs["guidance"])

            if "seed" in kwargs and kwargs["seed"] and kwargs["seed"] != -1:
                payload["seed"] = int(kwargs["seed"])

            print(f'🎨 Generating image with ModelScope: {model}')
            print(f'🎨 Prompt: {prompt}')
            print(f'🎨 Size: {payload["size"]}')

            # Send request
            response_data = await self._make_request(url, headers, payload)

            # Process response and save image
            result = await self._process_response(response_data, metadata)

            print(f'🎨 ModelScope image generation completed successfully')
            return result

        except Exception as e:
            print(f'❌ Error generating image with ModelScope: {e}')
            traceback.print_exc()
            raise e
