import traceback
from typing import Dict
from langchain_core.tools import BaseTool
from models.tool_model import ToolInfo
from tools.comfy_dynamic import build_tool
from tools.write_plan import write_plan_tool
from tools.generate_image_by_gpt_image_1_jaaz import generate_image_by_gpt_image_1_jaaz
from tools.generate_image_by_imagen_4_jaaz import generate_image_by_imagen_4_jaaz
from tools.generate_image_by_imagen_4_replicate import (
    generate_image_by_imagen_4_replicate,
)
from tools.generate_image_by_ideogram3_bal_jaaz import (
    generate_image_by_ideogram3_bal_jaaz,
)

# from tools.generate_image_by_flux_1_1_pro import generate_image_by_flux_1_1_pro
from tools.generate_image_by_flux_kontext_pro_jaaz import (
    generate_image_by_flux_kontext_pro_jaaz,
)
from tools.generate_image_by_flux_kontext_pro_replicate import (
    generate_image_by_flux_kontext_pro_replicate,
)
from tools.generate_image_by_flux_kontext_max_jaaz import (
    generate_image_by_flux_kontext_max,
)
from tools.generate_image_by_flux_kontext_max_replicate import (
    generate_image_by_flux_kontext_max_replicate,
)
from tools.generate_image_by_doubao_seedream_3_jaaz import (
    generate_image_by_doubao_seedream_3_jaaz,
)
from tools.generate_image_by_doubao_seedream_3_volces import (
    generate_image_by_doubao_seedream_3_volces,
)
from tools.generate_image_by_doubao_seededit_3_volces import (
    edit_image_by_doubao_seededit_3_volces,
)
from tools.generate_video_by_seedance_v1_jaaz import generate_video_by_seedance_v1_jaaz
from tools.generate_video_by_seedance_v1_pro_volces import (
    generate_video_by_seedance_v1_pro_volces,
)
from tools.generate_video_by_seedance_v1_lite_volces import (
    generate_video_by_seedance_v1_lite_t2v,
    generate_video_by_seedance_v1_lite_i2v,
)
from tools.generate_video_by_kling_v2_jaaz import generate_video_by_kling_v2_jaaz
from tools.generate_image_by_recraft_v3_jaaz import generate_image_by_recraft_v3_jaaz
from tools.generate_image_by_recraft_v3_replicate import (
    generate_image_by_recraft_v3_replicate,
)
from tools.generate_video_by_hailuo_02_jaaz import generate_video_by_hailuo_02_jaaz
from tools.generate_video_by_veo3_fast_jaaz import generate_video_by_veo3_fast_jaaz
<<<<<<< Updated upstream
from tools.generate_image_by_midjourney import generate_image_by_midjourney
=======
from tools.generate_image_by_midjourney_jaaz import generate_image_by_midjourney_jaaz
from tools.analyze_video_by_gemini import analyze_video_by_gemini

# ModelScope tools are now dynamically registered through configuration
>>>>>>> Stashed changes
from services.config_service import config_service
from services.db_service import db_service

TOOL_MAPPING: Dict[str, ToolInfo] = {
    "generate_image_by_gpt_image_1_jaaz": {
        "display_name": "GPT Image 1",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_gpt_image_1_jaaz,
    },
    "generate_image_by_imagen_4_jaaz": {
        "display_name": "Imagen 4",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_imagen_4_jaaz,
    },
    "generate_image_by_recraft_v3_jaaz": {
        "display_name": "Recraft v3",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_recraft_v3_jaaz,
    },
    "generate_image_by_ideogram3_bal_jaaz": {
        "display_name": "Ideogram 3 Balanced",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_ideogram3_bal_jaaz,
    },
    # "generate_image_by_flux_1_1_pro_jaaz": {
    #     "display_name": "Flux 1.1 Pro",
    #     "type": "image",
    #     "provider": "jaaz",
    #     "tool_function": generate_image_by_flux_1_1_pro,
    # },
    "generate_image_by_flux_kontext_pro_jaaz": {
        "display_name": "Flux Kontext Pro",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_flux_kontext_pro_jaaz,
    },
    "generate_image_by_flux_kontext_max_jaaz": {
        "display_name": "Flux Kontext Max",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_flux_kontext_max,
    },
    "generate_image_by_doubao_seedream_3_jaaz": {
        "display_name": "Doubao Seedream 3",
        "type": "image",
        "provider": "jaaz",
        "tool_function": generate_image_by_doubao_seedream_3_jaaz,
    },
    "generate_image_by_doubao_seedream_3_volces": {
        "display_name": "Doubao Seedream 3 by volces",
        "type": "image",
        "provider": "volces",
        "tool_function": generate_image_by_doubao_seedream_3_volces,
    },
    "edit_image_by_doubao_seededit_3_volces": {
        "display_name": "Doubao Seededit 3 by volces",
        "type": "image",
        "provider": "volces",
        "tool_function": edit_image_by_doubao_seededit_3_volces,
    },
    "generate_video_by_seedance_v1_jaaz": {
        "display_name": "Doubao Seedance v1",
        "type": "video",
        "provider": "jaaz",
        "tool_function": generate_video_by_seedance_v1_jaaz,
    },
    "generate_video_by_hailuo_02_jaaz": {
        "display_name": "Hailuo 02",
        "type": "video",
        "provider": "jaaz",
        "tool_function": generate_video_by_hailuo_02_jaaz,
    },
    "generate_video_by_kling_v2_jaaz": {
        "display_name": "Kling v2.1 Standard",
        "type": "video",
        "provider": "jaaz",
        "tool_function": generate_video_by_kling_v2_jaaz,
    },
    "generate_video_by_seedance_v1_pro_volces": {
        "display_name": "Doubao Seedance v1 by volces",
        "type": "video",
        "provider": "volces",
        "tool_function": generate_video_by_seedance_v1_pro_volces,
    },
    "generate_video_by_seedance_v1_lite_volces_t2v": {
        "display_name": "Doubao Seedance v1 lite(text-to-video)",
        "type": "video",
        "provider": "volces",
        "tool_function": generate_video_by_seedance_v1_lite_t2v,
    },
    "generate_video_by_seedance_v1_lite_i2v_volces": {
        "display_name": "Doubao Seedance v1 lite(images-to-video)",
        "type": "video",
        "provider": "volces",
        "tool_function": generate_video_by_seedance_v1_lite_i2v,
    },
    "generate_video_by_veo3_fast_jaaz": {
        "display_name": "Veo3 Fast",
        "type": "video",
        "provider": "jaaz",
        "tool_function": generate_video_by_veo3_fast_jaaz,
    },
    # ---------------
    # Replicate Tools
    # ---------------
    "generate_image_by_imagen_4_replicate": {
        "display_name": "Imagen 4",
        "type": "image",
        "provider": "replicate",
        "tool_function": generate_image_by_imagen_4_replicate,
    },
    "generate_image_by_recraft_v3_replicate": {
        "display_name": "Recraft v3",
        "type": "image",
        "provider": "replicate",
        "tool_function": generate_image_by_recraft_v3_replicate,
    },
    "generate_image_by_flux_kontext_pro_replicate": {
        "display_name": "Flux Kontext Pro",
        "type": "image",
        "provider": "replicate",
        "tool_function": generate_image_by_flux_kontext_pro_replicate,
    },
    "generate_image_by_flux_kontext_max_replicate": {
        "display_name": "Flux Kontext Max",
        "type": "image",
        "provider": "replicate",
        "tool_function": generate_image_by_flux_kontext_max_replicate,
    },
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    "generate_image_by_midjourney": {
        "display_name": "Midjourney",
        "type": "image",
        "provider": "midjourney",  
        "tool_function": generate_image_by_midjourney,
=======
=======
>>>>>>> Stashed changes
    "analyze_video_by_gemini": {
        "display_name": "Video Analysis by Gemini",
        "type": "analysis",
        "provider": "google",
        "tool_function": analyze_video_by_gemini,
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    },
}


class ToolService:
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self._register_required_tools()

    def _register_required_tools(self):
        """注册必须的工具"""
        try:
            self.tools["write_plan"] = {
                "provider": "system",
                "tool_function": write_plan_tool,
            }
        except ImportError as e:
            print(f"❌ 注册必须工具失败 write_plan: {e}")

    def register_tool(self, tool_id: str, tool_info: ToolInfo):
        """注册单个工具"""
        if tool_id in self.tools:
            print(f"🔄 TOOL ALREADY REGISTERED: {tool_id}")
            return

        self.tools[tool_id] = tool_info

    # TODO: Check if there will be racing conditions when server just starting up but tools are not ready yet.
    async def initialize(self):
        self.clear_tools()
        try:
            for provider_name, provider_config in config_service.app_config.items():
                # register all tools by api provider with api key OR for special providers (comfyui, midjourney)
                has_api_key = provider_config.get("api_key", "")
                has_url = provider_config.get("url", "")
                is_special_provider = provider_name in ['comfyui', 'midjourney']
                
                if has_api_key or (is_special_provider and has_url):
                    for tool_id, tool_info in TOOL_MAPPING.items():
                        if tool_info.get("provider") == provider_name:
                            self.register_tool(tool_id, tool_info)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes

                    # For ModelScope: register a dynamic tool for each configured model
                    if provider_name == 'modelscope':
                        ms_models = provider_config.get('models') or {}
                        for model_name in ms_models.keys():
                            dynamic_tool_id = f"modelscope__{model_name.replace('/', '_').replace('-', '_')}"
                            if dynamic_tool_id in self.tools:
                                continue
                            from typing import Annotated, Optional
                            from pydantic import BaseModel, Field  # type: ignore
                            from langchain_core.tools import tool, InjectedToolCallId  # type: ignore
                            from langchain_core.runnables import RunnableConfig  # type: ignore

                            class _FixedSchema(BaseModel):
                                prompt: str = Field(
                                    description="Required. Detailed English text prompt describing the image to generate (e.g. 'A realistic illustration of a British Shorthair cat, sitting upright and gazing directly at the viewer')"
                                )
                                model: str = Field(
                                    default=model_name,
                                    description=f"Required. ModelScope model ID: {model_name}",
                                )
                                aspect_ratio: str = Field(
                                    default="1:1",
                                    description="Required. Image aspect ratio - must be one of: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3",
                                )
                                negative_prompt: Optional[str] = Field(
                                    default=None,
                                    description="Optional. Text describing what should NOT appear in the image",
                                )
                                steps: Optional[int] = Field(
                                    default=30,
                                    description="Optional. Number of sampling steps for image generation (1-100)",
                                )
                                guidance: Optional[float] = Field(
                                    default=3.5,
                                    description="Optional. Guidance scale for prompt adherence (1.5-20)",
                                )
                                seed: Optional[int] = Field(
                                    default=None,
                                    description="Optional. Random seed for reproducible results",
                                )
                                tool_call_id: Annotated[str, InjectedToolCallId]

                            @tool(
                                dynamic_tool_id,
                                description=f"Generate image using ModelScope model: {model_name}. This tool creates AI-generated images from text prompts using the specified ModelScope model. Required parameters: prompt (descriptive text), aspect_ratio (1:1, 16:9, 9:16, etc.). Optional: negative_prompt, steps (1-100), guidance (1.5-20), seed.",
                                args_schema=_FixedSchema,
                            )
                            async def _run(
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
                                print(
                                    f"🛠️ ModelScope工具被调用，tool_call_id: {tool_call_id}"
                                )
                                print(
                                    f"🛠️ 参数: prompt={prompt}, model={model}, aspect_ratio={aspect_ratio}, steps={steps}, guidance={guidance}, seed={seed}"
                                )

                                from tools.generate_image_by_modelscope import (
                                    generate_image_by_modelscope as base,
                                )

                                return await base(
                                    prompt=prompt,
                                    model=model,
                                    aspect_ratio=aspect_ratio,
                                    config=config,
                                    tool_call_id=tool_call_id,
                                    negative_prompt=negative_prompt,
                                    steps=steps,
                                    guidance=guidance,
                                    seed=seed,
                                )

                            print(f"🛠️ 注册ModelScope动态工具: {dynamic_tool_id}")
                            print(f"🛠️ 工具函数类型: {type(_run)}")
                            self.register_tool(
                                dynamic_tool_id,
                                {
                                    "provider": "modelscope",
                                    "display_name": model_name,
                                    "type": "image",
                                    "tool_function": _run,  # type: ignore
                                },
                            )
            # Register comfyui workflow tools
            if config_service.app_config.get("comfyui", {}).get("url", ""):
                await register_comfy_tools()
        except Exception as e:
            print(f"❌ Failed to initialize tool service: {e}")
            traceback.print_stack()

<<<<<<< Updated upstream
=======
    async def _register_dynamic_image_models(
        self, provider_name: str, provider_config: dict
    ):
        """Register dynamic image generation tools based on user-configured models"""
        try:
            # Skip ModelScope - it has its own dedicated registration logic
            if provider_name == "modelscope":
                return

            models = provider_config.get("models", {})
            for model_id, model_info in models.items():
                if model_info.get("type") == "image":
                    # Create dynamic tool ID
                    safe_model_name = (
                        model_id.replace("/", "_").replace("-", "_").lower()
                    )
                    tool_id = f"generate_image_by_{provider_name}_{safe_model_name}"

                    # Skip if tool already exists (avoid duplicates)
                    if tool_id in self.tools:
                        continue

                    # Create display name
                    display_name = f"{provider_name.title()} {model_id}"

                    # For other providers (not ModelScope), add additional logic here if needed
                    print(
                        f"ℹ️ Skipping dynamic registration for {provider_name} - not implemented yet"
                    )

        except Exception as e:
            print(
                f"❌ Failed to register dynamic image models for {provider_name}: {e}"
            )
            import traceback

            traceback.print_exc()

>>>>>>> Stashed changes
    def get_tool(self, tool_name: str) -> BaseTool | None:
        tool_info = self.tools.get(tool_name)
        return tool_info.get("tool_function") if tool_info else None

    def remove_tool(self, tool_id: str):
        self.tools.pop(tool_id)

    def get_all_tools(self) -> Dict[str, ToolInfo]:
        return self.tools.copy()

    def clear_tools(self):
        self.tools.clear()
        # 重新注册必须的工具
        self._register_required_tools()


tool_service = ToolService()


async def register_comfy_tools() -> Dict[str, BaseTool]:
    """
    Fetch all workflows from DB and build tool callables.
    Run inside the current event loop.
    """
    dynamic_comfy_tools: Dict[str, BaseTool] = {}
    try:
        workflows = await db_service.list_comfy_workflows()
    except Exception as exc:  # pragma: no cover
        print("[comfy_dynamic] Failed to list comfy workflows:", exc)
        traceback.print_stack()
        return {}

    for wf in workflows:
        try:
            tool_fn = build_tool(wf)
            # Export with a unique python identifier so that `dir(module)` works
            unique_name = f"comfyui_{wf['name']}"
            dynamic_comfy_tools[unique_name] = tool_fn
            tool_service.register_tool(
                unique_name,
                {
                    "provider": "comfyui",
                    "tool_function": tool_fn,
                    "display_name": wf["name"],
                    # TODO: Add comfyui workflow type! Not hardcoded!
                    "type": "image",
                },
            )
        except Exception as exc:  # pragma: no cover
            print(
                f"[comfy_dynamic] Failed to create tool for workflow {wf.get('id')}: {exc}"
            )
            print(traceback.print_stack())

    return dynamic_comfy_tools
