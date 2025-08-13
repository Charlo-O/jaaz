"""
Dynamic registration of ModelScope models as LangChain tools.

Similar to ComfyUI workflows, this module allows ModelScope models to be 
registered as callable tools with descriptions that help LLMs understand 
when and how to use them.

Usage:
    from server.tools.modelscope_dynamic import register_modelscope_tools
    tools = await register_modelscope_tools()
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Annotated, Any, Dict, Optional
from common import DEFAULT_PORT
from .utils.image_canvas_utils import (
    generate_file_id,
    generate_new_image_element,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from pydantic import BaseModel, Field
from services.config_service import config_service
from services.db_service import db_service
from services.websocket_service import broadcast_session_update, send_to_websocket
from .image_providers.modelscope_provider import ModelScopeImageProvider


class ModelScopeToolInputSchema(BaseModel):
    """Input schema for ModelScope image generation tools"""

    prompt: str = Field(
        description="Image generation prompt describing what you want to create"
    )
    aspect_ratio: str = Field(
        default="1:1",
        description="Image aspect ratio. Options: 1:1, 16:9, 9:16, 4:3, 3:4",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Negative prompt describing what you don't want in the image",
    )
    steps: Optional[int] = Field(
        default=30, description="Number of denoising steps (typically 20-50)"
    )
    guidance: Optional[float] = Field(
        default=7.5,
        description="Guidance scale for prompt adherence (typically 5.0-15.0)",
    )
    seed: Optional[int] = Field(
        default=-1, description="Random seed for reproducible results (-1 for random)"
    )
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        description="Tool call identifier"
    )


def build_modelscope_tool(model_name: str, model_config: Dict[str, Any]) -> BaseTool:
    """Create a LangChain tool for a specific ModelScope model"""

    # Get description from model config, or create a default one
    description = model_config.get(
        "description", f"Generate images using ModelScope model {model_name}"
    )

    @tool(
        description=description,
        args_schema=ModelScopeToolInputSchema,
    )
    async def _generate_image(
        config: RunnableConfig,
        tool_call_id: Annotated[str, InjectedToolCallId],
        prompt: str,
        aspect_ratio: str = "1:1",
        negative_prompt: Optional[str] = None,
        steps: int = 30,
        guidance: float = 7.5,
        seed: int = -1,
    ) -> str:
        """Generate image using ModelScope API"""
        print(f"🛠️ ModelScope tool_call_id: {tool_call_id}")

        ctx = config.get("configurable", {})
        canvas_id = ctx.get("canvas_id", "")
        session_id = ctx.get("session_id", "")
        print(f"🛠️ canvas_id: {canvas_id}, session_id: {session_id}")

        # Inject the tool call id into the context
        ctx["tool_call_id"] = tool_call_id

        try:
            # Create provider instance
            provider = ModelScopeImageProvider()

            # Prepare kwargs
            kwargs = {}
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt
            if steps:
                kwargs["steps"] = steps
            if guidance:
                kwargs["guidance"] = guidance
            if seed and seed != -1:
                kwargs["seed"] = seed

            print(f'🎨 Generating image with ModelScope model: {model_name}')
            print(f'🎨 Prompt: {prompt}')
            print(f'🎨 Aspect ratio: {aspect_ratio}')

            # Generate image
            mime_type, width, height, filename = await provider.generate(
                prompt=prompt, model=model_name, aspect_ratio=aspect_ratio, **kwargs
            )

            # Update canvas data
            canvas_data = await db_service.get_canvas_data(canvas_id)
            if canvas_data is None:
                canvas_data = {"data": {}}
            if "data" not in canvas_data:
                canvas_data["data"] = {}
            if "elements" not in canvas_data["data"]:
                canvas_data["data"]["elements"] = []
            if "files" not in canvas_data["data"]:
                canvas_data["data"]["files"] = {}

            # Generate file ID and create element
            file_id = generate_file_id()
            url = f"/api/file/{filename}"

            file_data = {
                "mimeType": mime_type,
                "id": file_id,
                "dataURL": url,
                "created": int(time.time() * 1000),
            }

            # Create new image element
            new_element = await generate_new_image_element(
                canvas_id,
                file_id,
                {
                    "width": width,
                    "height": height,
                },
                canvas_data=canvas_data.get("data", {}),
            )

            # Add to canvas data
            canvas_data["data"]["elements"].append(new_element)
            canvas_data["data"]["files"][file_id] = file_data

            # Save canvas data
            await db_service.save_canvas_data(
                canvas_id, json.dumps(canvas_data["data"])
            )

            # Broadcast update
            image_url = f"http://localhost:{DEFAULT_PORT}/api/file/{filename}"
            await broadcast_session_update(
                session_id,
                canvas_id,
                {
                    "type": "image_generated",
                    "element": new_element,
                    "file": file_data,
                    "image_url": image_url,
                },
            )

            # Return markdown
            markdown_image = f"![id: {filename}]({image_url})"
            return f"Successfully generated image using {model_name}: {markdown_image}"

        except Exception as e:
            error_msg = str(e)
            print(
                f"❌ Error generating image with ModelScope {model_name}: {error_msg}"
            )
            traceback.print_exc()

            # Send user-friendly error message
            await send_to_websocket(session_id, {"type": "error", "error": error_msg})
            return f"使用 {model_name} 生成图片失败: {error_msg}"

    # Set the tool name manually - just use the model name
    tool_name = model_name.replace('/', '_').replace('-', '_')
    _generate_image.name = tool_name

    return _generate_image


async def register_modelscope_tools() -> Dict[str, BaseTool]:
    """
    Register ModelScope models as dynamic tools based on configuration.

    Returns:
        Dict[str, BaseTool]: Dictionary of registered ModelScope tools
    """
    dynamic_modelscope_tools: Dict[str, BaseTool] = {}

    try:
        # Get ModelScope configuration
        modelscope_config = config_service.app_config.get('modelscope', {})
        models = modelscope_config.get('models', {})

        if not models:
            print("[modelscope_dynamic] No ModelScope models configured")
            return {}

        print(
            f"[modelscope_dynamic] Registering {len(models)} ModelScope models as tools"
        )

        for model_name, model_config in models.items():
            try:
                # Only register image models
                if model_config.get('type') != 'image':
                    continue

                tool_fn = build_modelscope_tool(model_name, model_config)

                # Use clean model name as tool ID
                clean_name = model_name.replace('/', '_').replace('-', '_')
                dynamic_modelscope_tools[clean_name] = tool_fn

                print(
                    f"[modelscope_dynamic] Registered tool: {clean_name} for model: {model_name}"
                )

            except Exception as exc:
                print(
                    f"[modelscope_dynamic] Failed to create tool for model {model_name}: {exc}"
                )
                traceback.print_exc()

        print(
            f"[modelscope_dynamic] Successfully registered {len(dynamic_modelscope_tools)} ModelScope tools"
        )

    except Exception as e:
        print(f"[modelscope_dynamic] Error registering ModelScope tools: {e}")
        traceback.print_exc()

    return dynamic_modelscope_tools
