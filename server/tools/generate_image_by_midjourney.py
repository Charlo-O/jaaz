from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByMidjourneyInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. Midjourney supports very detailed and creative prompts. You can include style modifiers like '--v 6.1' for version, '--stylize' for styling, etc."
    )
    aspect_ratio: str = Field(
        description="Required. Aspect ratio of the image, only these values are allowed: 1:1, 16:9, 4:3, 3:4, 9:16. Choose the best fitting aspect ratio according to the prompt."
    )
    enable_upscale: bool = Field(
        default=True,
        description="Optional. If True, will generate 4-grid image and then upscale all 4 images with 5s intervals. If False, only returns the 4-grid image."
    )
    input_images: Optional[list[str]] = Field(
        default=None,
        description="Optional. List of input image file paths to use as reference or for image variations. These images will be sent as base64 to Midjourney."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_image_by_midjourney",
      description="Generate an image using Midjourney AI via Proxy API. Excellent for creative, artistic, and detailed image generation. Supports input images for reference, variations, and blending. Known for producing high-quality, imaginative artwork. Can generate 4-grid image or upscale all 4 images individually.",
      args_schema=GenerateImageByMidjourneyInputSchema)
async def generate_image_by_midjourney(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    enable_upscale: bool = True,
    input_images: Optional[list[str]] = None,
) -> str:
    """
    Generate an image using Midjourney via Proxy API
    
    Args:
        prompt: Image generation prompt (supports Midjourney-specific parameters)
        aspect_ratio: Image aspect ratio (1:1, 16:9, 4:3, 3:4, 9:16)
        config: Configuration context from langgraph
        tool_call_id: Tool call identifier
        enable_upscale: If True, upscale all 4 images with 5s intervals
        input_images: Optional list of reference image paths
        
    Returns:
        str: Generation result message with image URL(s)
    """
    ctx = config.get('configurable', {})
    canvas_id = ctx.get('canvas_id', '')
    session_id = ctx.get('session_id', '')

    if enable_upscale:
        # Use the new upscale functionality
        return await generate_image_with_provider(
            canvas_id=canvas_id,
            session_id=session_id,
            provider='midjourney',
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            model="midjourney",
            input_images=input_images,
            use_upscale=True,  # This will be passed to the provider
        )
    else:
        # Use original 4-grid only functionality
        return await generate_image_with_provider(
            canvas_id=canvas_id,
            session_id=session_id,
            provider='midjourney',
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            model="midjourney",
            input_images=input_images,
        )


# Export the tool for easy import
__all__ = ["generate_image_by_midjourney"]