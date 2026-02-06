from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool  # type: ignore
from pydantic import BaseModel, Field

from tools.utils.image_generation_core import generate_image_with_provider


class GenerateImageByMJProxyInputSchema(BaseModel):
    prompt: str = Field(
        description=(
            "Required. The prompt for image generation. You may include Midjourney prompt flags "
            "(e.g. --stylize, --chaos). Version/model flags are handled by the selected tool (V7/V6/Niji)."
        )
    )
    aspect_ratio: str = Field(
        description=(
            "Required. Aspect ratio of the image, only these values are allowed: "
            "1:1, 16:9, 4:3, 3:4, 9:16. Choose the best fitting aspect ratio according to the prompt."
        )
    )
    input_images: list[str] | None = Field(
        default=None,
        description=(
            "Optional. Placeholder images as references. Pass image_id list, e.g. ['im_jurheut7.png']. "
            "These will be sent to MJAPI as base64Array."
        ),
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "generate_image_by_midjourney_v7_mjproxy",
    description=(
        "Generate images via local MJProxy/MJAPI using Midjourney V7. "
        "Uses MID_JOURNEY botType and appends --v 7 when not already present."
    ),
    args_schema=GenerateImageByMJProxyInputSchema,
)
async def generate_image_by_midjourney_v7_mjproxy(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    input_images: list[str] | None = None,
) -> str:
    ctx = config.get("configurable", {})
    canvas_id = ctx.get("canvas_id", "")
    session_id = ctx.get("session_id", "")

    return await generate_image_with_provider(
        canvas_id=canvas_id,
        session_id=session_id,
        provider="mjproxy",
        model="mj_v7",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        input_images=input_images,
    )


@tool(
    "generate_image_by_midjourney_v6_mjproxy",
    description=(
        "Generate images via local MJProxy/MJAPI using Midjourney V6. "
        "Uses MID_JOURNEY botType and appends --v 6 when not already present."
    ),
    args_schema=GenerateImageByMJProxyInputSchema,
)
async def generate_image_by_midjourney_v6_mjproxy(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    input_images: list[str] | None = None,
) -> str:
    ctx = config.get("configurable", {})
    canvas_id = ctx.get("canvas_id", "")
    session_id = ctx.get("session_id", "")

    return await generate_image_with_provider(
        canvas_id=canvas_id,
        session_id=session_id,
        provider="mjproxy",
        model="mj_v6",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        input_images=input_images,
    )


@tool(
    "generate_image_by_niji_mjproxy",
    description=(
        "Generate images via local MJProxy/MJAPI using Niji Journey (anime style). "
        "Uses NIJI_JOURNEY botType."
    ),
    args_schema=GenerateImageByMJProxyInputSchema,
)
async def generate_image_by_niji_mjproxy(
    prompt: str,
    aspect_ratio: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    input_images: list[str] | None = None,
) -> str:
    ctx = config.get("configurable", {})
    canvas_id = ctx.get("canvas_id", "")
    session_id = ctx.get("session_id", "")

    return await generate_image_with_provider(
        canvas_id=canvas_id,
        session_id=session_id,
        provider="mjproxy",
        model="niji",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        input_images=input_images,
    )


__all__ = [
    "generate_image_by_midjourney_v7_mjproxy",
    "generate_image_by_midjourney_v6_mjproxy",
    "generate_image_by_niji_mjproxy",
]
