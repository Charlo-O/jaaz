import asyncio
import os
import traceback
from typing import Any, Optional

from services.config_service import FILES_DIR, config_service
from utils.http_client import HttpClient

from ..utils.image_utils import generate_image_id, get_image_info_and_save
from .image_base_provider import ImageProviderBase


class MJProxyImageProvider(ImageProviderBase):
    """MJProxy / MJAPI image generation provider.

    Based on `mjapiswagger.json`:
    - POST /mj/submit/imagine (SubmitImagineDTO)
    - GET  /mj/task/{id}/fetch (TaskInfo)
    - Auth: Authorization: {Token}
    """

    _DEFAULT_TIMEOUT_SECONDS = 10 * 60
    _DEFAULT_POLL_INTERVAL_SECONDS = 2

    def _get_base_url(self) -> str:
        config = config_service.app_config.get("mjproxy", {})
        base_url = str(config.get("url", "")).strip().rstrip("/")
        if not base_url:
            raise ValueError("mjproxy URL is not configured")
        return base_url

    def _build_headers(self) -> dict[str, str]:
        config = config_service.app_config.get("mjproxy", {})
        api_key = str(config.get("api_key", "")).strip()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        # Swagger defines apiKey in header "Authorization".
        if api_key:
            headers["Authorization"] = api_key
        return headers

    def _resolve_model_preset(self, model: str) -> dict[str, Optional[str]]:
        """Map our internal model id -> MJAPI request knobs."""

        normalized = (model or "").strip().lower()
        if normalized in {"mj_v7", "midjourney_v7", "midjourney-v7", "v7", "mj7"}:
            return {
                "botType": "MID_JOURNEY",
                "versionFlag": "--v 7",
            }
        if normalized in {"mj_v6", "midjourney_v6", "midjourney-v6", "v6", "mj6"}:
            return {
                "botType": "MID_JOURNEY",
                "versionFlag": "--v 6",
            }
        if normalized in {"niji", "niji_journey", "niji-journey"}:
            return {
                "botType": "NIJI_JOURNEY",
                "versionFlag": None,
            }

        # Default: Midjourney bot, no explicit version flag.
        return {
            "botType": "MID_JOURNEY",
            "versionFlag": None,
        }

    def _ensure_suffix_flag(self, prompt: str, flag: Optional[str]) -> str:
        if not flag:
            return prompt
        if flag in prompt:
            return prompt
        # Avoid adding multiple version flags.
        if flag.startswith("--v") and ("--v " in prompt or "--version " in prompt):
            return prompt
        return f"{prompt.rstrip()} {flag}".strip()

    def _ensure_aspect_ratio(self, prompt: str, aspect_ratio: str) -> str:
        # Let user override explicitly.
        lowered = prompt.lower()
        if " --ar " in lowered or " --aspect " in lowered:
            return prompt
        ratio = (aspect_ratio or "").strip()
        if not ratio or ratio == "1:1":
            return prompt
        return f"{prompt.rstrip()} --ar {ratio}".strip()

    async def _submit_imagine(
        self,
        prompt: str,
        bot_type: Optional[str],
        base64_array: Optional[list[str]],
    ) -> str:
        url = f"{self._get_base_url()}/mj/submit/imagine"
        headers = self._build_headers()

        payload: dict[str, Any] = {
            "prompt": prompt,
        }
        if bot_type:
            payload["botType"] = bot_type
        if base64_array:
            payload["base64Array"] = base64_array

        async with HttpClient.create_aiohttp() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(
                        f"mjproxy imagine failed: HTTP {response.status} - {await response.text()}"
                    )
                res = await response.json()

        task_id = res.get("result")
        if isinstance(task_id, str) and task_id.strip():
            return task_id.strip()

        description = res.get("description") or "unknown error"
        code = res.get("code")
        raise Exception(
            f"mjproxy imagine failed: code={code} description={description}"
        )

    async def _fetch_task(self, task_id: str) -> dict[str, Any]:
        url = f"{self._get_base_url()}/mj/task/{task_id}/fetch"
        headers = self._build_headers()
        async with HttpClient.create_aiohttp() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(
                        f"mjproxy task fetch failed: HTTP {response.status} - {await response.text()}"
                    )
                return await response.json()

    async def _submit_action(
        self,
        task_id: str,
        custom_id: str,
        *,
        strong: Optional[bool] = None,
        enable_remix: Optional[bool] = None,
    ) -> str:
        """Submit an action for a task (e.g. Upscale U1).

        MJAPI uses `customId` to identify which button/action to run.
        """

        url = f"{self._get_base_url()}/mj/submit/action"
        headers = self._build_headers()

        payload: dict[str, Any] = {
            "taskId": task_id,
            "customId": custom_id,
        }
        if strong is not None:
            payload["strong"] = strong
        if enable_remix is not None:
            payload["enableRemix"] = enable_remix

        async with HttpClient.create_aiohttp() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(
                        f"mjproxy submit action failed: HTTP {response.status} - {await response.text()}"
                    )
                res = await response.json()

        next_task_id = res.get("result")
        if isinstance(next_task_id, str) and next_task_id.strip():
            return next_task_id.strip()

        description = res.get("description") or "unknown error"
        code = res.get("code")
        raise Exception(
            f"mjproxy submit action failed: code={code} description={description}"
        )

    async def _wait_for_task_completion(
        self,
        task_id: str,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        while True:
            task = await self._fetch_task(task_id)
            status = task.get("status")

            if status == "SUCCESS":
                return task
            if status in {"FAILURE", "CANCEL"}:
                raise Exception(
                    task.get("failReason") or f"Task {task_id} failed: {status}"
                )
            if status == "MODAL":
                raise Exception(
                    task.get("failReason")
                    or "Task requires additional confirmation/verification (MODAL)"
                )

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Timeout waiting for mjproxy task {task_id} ({timeout_seconds}s)"
                )

            await asyncio.sleep(poll_interval_seconds)

    def _get_button_custom_id(self, task: dict[str, Any], label: str) -> Optional[str]:
        buttons = task.get("buttons")
        if not isinstance(buttons, list):
            return None

        for b in buttons:
            if not isinstance(b, dict):
                continue
            b_label = str(b.get("label", "")).strip()
            if b_label.upper() == label.strip().upper():
                custom_id = b.get("customId")
                if isinstance(custom_id, str) and custom_id.strip():
                    return custom_id.strip()
        return None

    def _get_first_upsample_custom_id(self, task: dict[str, Any]) -> Optional[str]:
        """Try to locate the Upscale(U1) action from TaskInfo.buttons."""

        # Prefer explicit U1 labels.
        for lbl in ["U1", "Upscale 1", "Upscale (1)"]:
            custom_id = self._get_button_custom_id(task, lbl)
            if custom_id:
                return custom_id

        # Fallback: search customId pattern.
        buttons = task.get("buttons")
        if not isinstance(buttons, list):
            return None

        for b in buttons:
            if not isinstance(b, dict):
                continue
            custom_id = b.get("customId")
            if not isinstance(custom_id, str):
                continue

            cid = custom_id.lower()
            if "upsample::1::" in cid or "upscale::1::" in cid:
                return custom_id.strip()

        return None

    def _extract_image_url(self, task: dict[str, Any]) -> str:
        image_url = task.get("imageUrl")
        if isinstance(image_url, str) and image_url.strip():
            return image_url

        image_urls = task.get("imageUrls")
        if isinstance(image_urls, list) and len(image_urls) > 0:
            first = image_urls[0]
            if isinstance(first, dict):
                url = first.get("url")
                if isinstance(url, str) and url.strip():
                    return url

        raise Exception("No imageUrl found in mjproxy task result")

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[str, int, int, str]:
        """Generate an image via MJProxy and return local filename info."""

        try:
            preset = self._resolve_model_preset(model)
            bot_type = preset.get("botType")

            final_prompt = prompt or ""
            final_prompt = self._ensure_aspect_ratio(final_prompt, aspect_ratio)
            final_prompt = self._ensure_suffix_flag(
                final_prompt, preset.get("versionFlag")
            )

            imagine_task_id = await self._submit_imagine(
                prompt=final_prompt,
                bot_type=bot_type,
                base64_array=input_images,
            )

            task = await self._wait_for_task_completion(imagine_task_id)

            # Midjourney initial IMAGINE usually returns a 2x2 grid. If U1 is
            # available, automatically upscale the first image.
            upsample_task_id: Optional[str] = None
            if task.get("action") == "IMAGINE":
                u1_custom_id = self._get_first_upsample_custom_id(task)
                if u1_custom_id:
                    upsample_task_id = await self._submit_action(
                        task_id=imagine_task_id,
                        custom_id=u1_custom_id,
                    )
                    task = await self._wait_for_task_completion(upsample_task_id)

            image_url = self._extract_image_url(task)

            image_id = generate_image_id()
            mime_type, width, height, extension = await get_image_info_and_save(
                image_url,
                os.path.join(FILES_DIR, f"{image_id}"),
                metadata={
                    **(metadata or {}),
                    "mjproxy_task_id": imagine_task_id,
                    "mjproxy_upsample_task_id": upsample_task_id,
                    "mjproxy_status": task.get("status"),
                    "mjproxy_botType": bot_type,
                    "mjproxy_promptFull": task.get("promptFull"),
                },
            )

            filename = f"{image_id}.{extension}"
            return mime_type, width, height, filename

        except Exception as e:
            print("Error generating image with MJProxy:", e)
            traceback.print_exc()
            raise
