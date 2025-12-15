from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByModelScopeInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for image generation. Describe what you want to generate in detail."
    )
    aspect_ratio: str = Field(
        description="Required. Aspect ratio of the image, only these values are allowed: 1:1, 16:9, 4:3, 3:4, 9:16. Choose the best fitting aspect ratio according to the prompt. Best ratio for posters is 3:4"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


def build_modelscope_tool(model_name: str):
    """
    Dynamically build a ModelScope tool for a specific model
    """
    # Clean model name for tool name (letters, numbers, underscores only)
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    tool_name = f"generate_image_by_modelscope_{safe_name}"
    
    async def run_tool(
        prompt: str,
        aspect_ratio: str,
        config: RunnableConfig,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        ctx = config.get('configurable', {})
        canvas_id = ctx.get('canvas_id', '')
        session_id = ctx.get('session_id', '')

        return await generate_image_with_provider(
            canvas_id=canvas_id,
            session_id=session_id,
            provider='modelscope',
            # Add provider prefix if not present, as expected by provider logic
            model=f"modelscope/{model_name}" if not model_name.startswith("modelscope/") else model_name,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            input_images=None,
        )

    return StructuredTool.from_function(
        coroutine=run_tool,
        name=tool_name,
        description=f"Generate image using ModelScope model: {model_name}",
        args_schema=GenerateImageByModelScopeInputSchema,
    )
