from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
from langchain_core.runnables import RunnableConfig
from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByModelScopeInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. Use descriptive English prompts for better results."
    )
    model: str = Field(
        description="Required. ModelScope model ID to use for generation (e.g., 'MAILAND/majicflus_v1', 'AI-ModelScope/stable-diffusion-v1-5', 'Qwen/Qwen-Image'). Default: 'MAILAND/majicflus_v1'"
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
        description="Optional. Number of sampling steps (1-100)"
    )
    guidance: Optional[float] = Field(
        default=3.5,
        description="Optional. Guidance scale for prompt adherence (1.5-20)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional. Random seed for reproducible results"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool("generate_image_by_modelscope",
      description="Generate an image using ModelScope API. Supports various open-source models like MAILAND/majicflus_v1 (default), Stable Diffusion, and others hosted on ModelScope platform. Good for text-to-image generation with customizable parameters.",
      args_schema=GenerateImageByModelScopeInputSchema)
async def generate_image_by_modelscope(
    prompt: str,
    model: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = 30,
    guidance: Optional[float] = 3.5,
    seed: Optional[int] = None,
) -> str:
    """
    Generate an image using ModelScope API
    
    Args:
        prompt: Image generation prompt
        model: ModelScope model ID (default: MAILAND/majicflus_v1)
        aspect_ratio: Image aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
        config: Configuration context from langgraph
        tool_call_id: Tool call identifier
        negative_prompt: Optional negative prompt
        steps: Number of sampling steps
        guidance: Guidance scale
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
        model=model,
        input_images=None,  # ModelScope text-to-image doesn't support input images
        **kwargs
    )


# Export the tool for easy import
__all__ = ["generate_image_by_modelscope"]