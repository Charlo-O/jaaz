"""
ModelScope (魔搭) image generation provider implementation
API Documentation: https://www.modelscope.cn/docs/model-service/API-Inference/intro
"""

import os
import traceback
from types import NoneType
from typing import Optional, Any
from .image_base_provider import ImageProviderBase
from ..utils.image_utils import get_image_info_and_save, generate_image_id
from services.config_service import FILES_DIR, config_service
from utils.http_client import HttpClient


class ModelScopeProvider(ImageProviderBase):
    """ModelScope (魔搭) image generation provider implementation"""

    def _calculate_dimensions(self, aspect_ratio: str) -> tuple[int, int]:
        """Calculate width and height based on aspect ratio"""
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
        factor = (1024**2 / (w_ratio * h_ratio)) ** 0.5

        width = int((factor * w_ratio) / 64) * 64
        height = int((factor * h_ratio) / 64) * 64

        return width, height

    async def _process_response(
        self, image_url: str, error_prefix: str = "ModelScope"
    ) -> tuple[str, int, int, str]:
        """
        Process response and save image

        Args:
            image_url: Image URL or base64 data
            error_prefix: Error message prefix

        Returns:
            tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        if not image_url:
            raise Exception(
                f"{error_prefix} image generation failed: No valid image data in response"
            )

        image_id = generate_image_id()
        
        # Check if it's base64 data
        is_b64 = image_url.startswith("data:") or not image_url.startswith("http")
        if image_url.startswith("data:"):
            # Extract base64 data from data URL
            image_url = image_url.split(",", 1)[1] if "," in image_url else image_url
        
        mime_type, width, height, extension = await get_image_info_and_save(
            image_url, os.path.join(FILES_DIR, f"{image_id}"), is_b64=is_b64
        )

        filename = f"{image_id}.{extension}"
        return mime_type, width, height, filename

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: list[str] | NoneType = None,
        **kwargs: Any,
    ) -> tuple[str, int, int, str]:
        """
        Generate image using ModelScope API service

        Args:
            prompt: Image generation prompt
            model: Model name to use for generation
            aspect_ratio: Image aspect ratio (1:1, 16:9, 4:3, 3:4, 9:16)
            input_images: Optional input images for reference or editing
            **kwargs: Additional provider-specific parameters

        Returns:
            tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        try:
            # Remove provider prefix from model name
            model = model.replace("modelscope/", "")

            config = config_service.app_config.get("modelscope", {})
            api_key = str(config.get("api_key", ""))
            api_url = str(config.get("url", "https://api-inference.modelscope.cn/v1/"))

            if not api_key:
                raise ValueError("ModelScope API key is not configured")

            width, height = self._calculate_dimensions(aspect_ratio)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            # ModelScope OpenAI-compatible API format
            # https://www.modelscope.cn/docs/model-service/API-Inference/intro
            # Size format: "1024x1024" (use x)
            size_str = f"{width}x{height}"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "size": size_str,
                "n": 1,
            }

            # Add input image if provided (for image editing)
            if input_images and len(input_images) > 0:
                payload["image"] = input_images[0]

            url = str(api_url).rstrip("/") + "/images/generations"
            
            print(f"ModelScope API URL: {url}")
            print(f"ModelScope API Model: {model}")
            print(f"ModelScope API Size: {size_str}")

            async with HttpClient.create_aiohttp() as session:
                async with session.post(
                    url, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            print(f"ModelScope API Error Response: {error_data}")
                            error_message = error_data.get(
                                "error", {}).get("message", f"HTTP {response.status}")
                            if isinstance(error_message, dict):
                                error_message = str(error_data.get("error", f"HTTP {response.status}"))
                        except Exception:
                            error_text = await response.text()
                            print(f"ModelScope API Error Text: {error_text}")
                            error_message = f"HTTP {response.status}: {error_text[:200]}"
                        raise Exception(
                            f"ModelScope image generation failed: {error_message}"
                        )

                    result_dict = await response.json()
                    print(f"ModelScope API Response: {result_dict}")
                    
                    # Handle response format
                    # Format 1: OpenAI compatible format
                    if "data" in result_dict and len(result_dict["data"]) > 0:
                        image_data = result_dict["data"][0]
                        if "url" in image_data:
                            image_url = image_data["url"]
                        elif "b64_json" in image_data:
                            image_url = image_data["b64_json"]
                        else:
                            raise Exception("No valid image data in response")
                    # Format 2: Tongyi-MAI/Z-Image-Turbo format
                    elif "images" in result_dict and len(result_dict["images"]) > 0:
                        image_data = result_dict["images"][0]
                        if isinstance(image_data, dict) and "url" in image_data:
                            image_url = image_data["url"]
                        elif isinstance(image_data, str):
                            image_url = image_data
                        else:
                            raise Exception("No valid image data in images list")
                    # Format 3: Output format
                    elif "output" in result_dict:
                        # Alternative response format
                        output = result_dict["output"]
                        if isinstance(output, str):
                            image_url = output
                        elif isinstance(output, dict) and "image_url" in output:
                            image_url = output["image_url"]
                        elif isinstance(output, list) and len(output) > 0:
                            image_url = output[0] if isinstance(output[0], str) else output[0].get("url", "")
                        else:
                            raise Exception("No valid image data in response")
                    else:
                        raise Exception("Unexpected response format from ModelScope API")

                    print(f"👇ModelScope Image URL: {image_url[:100]}...")

            return await self._process_response(image_url, "ModelScope")

        except Exception as e:
            print("Error generating image with ModelScope:", e)
            traceback.print_exc()
            raise e
