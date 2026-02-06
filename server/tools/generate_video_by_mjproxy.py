from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool  # type: ignore
from pydantic import BaseModel, Field

from tools.utils.image_utils import process_input_image
from tools.video_generation.video_canvas_utils import (
    process_video_result,
    send_video_start_notification,
)
from tools.video_providers.mjproxy_provider import MJProxyVideoProvider


class GenerateVideoByMJProxyInputSchema(BaseModel):
    prompt: str = Field(
        default="",
        description=(
            "Optional. Extra text prompt to guide the video. If you only want to animate the input image, leave it empty."
        ),
    )
    motion: str = Field(
        default="low",
        description="Optional. Motion strength. Suggested values: low, high.",
    )
    loop: bool = Field(
        default=False,
        description="Optional. Whether to loop the video.",
    )
    video_type: str | None = Field(
        default=None,
        description="Optional. MJAPI videoType, if your MJProxy server supports it (e.g. HD/SD).",
    )
    batch_size: int | None = Field(
        default=1,
        description=(
            "Optional. MJAPI batchSize (bs). Default is 1 (bs1) to generate a single video. "
            "If your MJProxy server ignores this field, it is safe to leave it as default."
        ),
    )
    input_images: list[str] = Field(
        description=(
            "Required. Images to animate. Pass a list of image_id here, e.g. ['im_jurheut7.png']. "
            "If you pass 2 images, the 2nd one will be used as endImage."
        )
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(
    "generate_video_by_mjproxy",
    description=(
        "Generate a short video via local MJProxy/MJAPI (Midjourney video). "
        "Requires at least 1 input image."
    ),
    args_schema=GenerateVideoByMJProxyInputSchema,
)
async def generate_video_by_mjproxy(
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    input_images: list[str],
    prompt: str = "",
    motion: str = "low",
    loop: bool = False,
    video_type: str | None = None,
    batch_size: int | None = 1,
) -> str:
    ctx = config.get("configurable", {})
    canvas_id = ctx.get("canvas_id", "")
    session_id = ctx.get("session_id", "")
    ctx["tool_call_id"] = tool_call_id

    if not input_images:
        raise ValueError("input_images is required and cannot be empty")

    await send_video_start_notification(
        session_id, "Starting MJProxy video generation..."
    )

    processed_images: list[str] = []
    for image_id in input_images[:2]:
        processed = await process_input_image(image_id)
        if not processed:
            raise ValueError(f"Failed to process input image: {image_id}")
        processed_images.append(processed)

    provider = MJProxyVideoProvider()
    video_url = await provider.generate(
        prompt=prompt,
        model="mj_video",
        input_images=processed_images,
        motion=motion,
        loop=loop,
        video_type=video_type,
        batch_size=batch_size,
    )

    return await process_video_result(
        video_url=video_url,
        session_id=session_id,
        canvas_id=canvas_id,
        provider_name="mjproxy",
    )


__all__ = ["generate_video_by_mjproxy"]
