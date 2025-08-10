from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByQwenImageInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. Use descriptive English prompts for better results. Qwen-Image works best with detailed, creative prompts."
    )
    aspect_ratio: str = Field(
        description="Required. Aspect ratio of the image, only these values are allowed: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3. Choose the best fitting aspect ratio according to the prompt."
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Optional. Negative prompt to guide what should NOT be generated"
    )
    steps: Optional[int] = Field(
        default=30,
        description="Optional. Number of sampling steps (1-100). Higher values for better quality."
    )
    guidance: Optional[float] = Field(
        default=7.5,
        description="Optional. Guidance scale for prompt adherence (1.5-20). Higher values for stronger prompt following."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional. Random seed for reproducible results"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_image_by_qwen_image",
      description="Generate an image using Qwen-Image model from Alibaba Cloud's ModelScope platform. Qwen-Image is a powerful multimodal AI model that excels at understanding and generating high-quality images. It supports various styles including photorealistic, artistic, anime, and abstract art. The model is particularly good at following detailed prompts and maintaining consistency in style and composition.",
      args_schema=GenerateImageByQwenImageInputSchema)
async def generate_image_by_qwen_image(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = 30,
    guidance: Optional[float] = 7.5,
    seed: Optional[int] = None,
) -> str:
    """
    Generate an image using Qwen-Image model from ModelScope
    
    Args:
        prompt: Image generation prompt (be descriptive for best results)
        aspect_ratio: Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
        config: Configuration context from langgraph
        tool_call_id: Tool call identifier
        negative_prompt: Optional negative prompt
        steps: Number of sampling steps (default: 30)
        guidance: Guidance scale (default: 7.5)
        seed: Random seed for reproducibility
        
    Returns:
        str: Generation result message with image URL
    """
    ctx = config.get('configurable', {})
    canvas_id = ctx.get('canvas_id', '')
    session_id = ctx.get('session_id', '')

    # Prepare optional parameters
    kwargs = {}
    if negative_prompt:
        kwargs['negative_prompt'] = negative_prompt
    if steps is not None:
        kwargs['steps'] = steps
    if guidance is not None:
        kwargs['guidance'] = guidance
    if seed is not None:
        kwargs['seed'] = seed

    return await generate_image_with_provider(
        canvas_id=canvas_id,
        session_id=session_id,
        provider='modelscope',
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        model='Qwen/Qwen-Image',  # 固定使用 Qwen-Image 模型
        input_images=None,  # Qwen-Image text-to-image 不支持输入图像
        **kwargs
    )


# Export the tool for easy import
__all__ = ["generate_image_by_qwen_image"]

