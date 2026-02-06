import asyncio
import traceback
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

from services.config_service import config_service
from utils.http_client import HttpClient

from .video_base_provider import VideoProviderBase


class MJProxyVideoProvider(VideoProviderBase, provider_name="mjproxy"):
    """MJProxy / MJAPI video generation provider.

    Based on `mjapiswagger.json`:
    - POST /mj/submit/video (SubmitVideoDTO)
    - GET  /mj/task/{id}/fetch (TaskInfo.videoUrl)
    - Auth: Authorization: {Token}
    """

    _DEFAULT_TIMEOUT_SECONDS = 15 * 60
    _DEFAULT_POLL_INTERVAL_SECONDS = 3

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
        if api_key:
            headers["Authorization"] = api_key
        return headers

    async def _submit_video(
        self,
        prompt: str,
        image: str,
        motion: Optional[str] = None,
        end_image: Optional[str] = None,
        loop: bool = False,
        action: str = "VIDEO",
        task_id: Optional[str] = None,
        index: Optional[int] = None,
        video_type: Optional[str] = None,
        batch_size: Optional[int] = 1,
    ) -> str:
        url = f"{self._get_base_url()}/mj/submit/video"
        headers = self._build_headers()

        payload: dict[str, Any] = {
            "prompt": prompt,
            "image": image,
            "loop": loop,
            # Explicitly set action to avoid server-side defaults.
            # For new generation use VIDEO; for extension use VIDEO_EXTEND with taskId/index.
            "action": action,
        }

        if task_id:
            payload["taskId"] = task_id
        if index is not None:
            payload["index"] = index
        if motion:
            payload["motion"] = motion
        if end_image:
            payload["endImage"] = end_image
        if video_type:
            payload["videoType"] = video_type
        # Default bs1
        if batch_size is not None:
            payload["batchSize"] = batch_size

        async with HttpClient.create_aiohttp() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(
                        f"mjproxy video submit failed: HTTP {response.status} - {await response.text()}"
                    )
                res = await response.json()

        task_id = res.get("result")
        if isinstance(task_id, str) and task_id.strip():
            return task_id.strip()

        description = res.get("description") or "unknown error"
        code = res.get("code")
        raise Exception(
            f"mjproxy video submit failed: code={code} description={description}"
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
        """Submit an action for an existing task.

        For MJ video tasks, the initial /mj/submit/video often returns a preview image
        plus a U1 button (customId like `video_virtual_upscale`) which needs to be
        executed to get the final mp4.
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

    def _get_u1_custom_id(self, task: dict[str, Any]) -> Optional[str]:
        # Prefer explicit U1 label.
        custom_id = self._get_button_custom_id(task, "U1")
        if custom_id:
            return custom_id

        buttons = task.get("buttons")
        if not isinstance(buttons, list):
            return None

        # Fallback: match common customId patterns.
        for b in buttons:
            if not isinstance(b, dict):
                continue
            cid = b.get("customId")
            if not isinstance(cid, str):
                continue
            lcid = cid.lower()
            if (
                "video_virtual_upscale::1::" in lcid
                or "upsample::1::" in lcid
                or "upscale::1::" in lcid
            ):
                return cid.strip()

        return None

    def _looks_like_video_url(self, value: str) -> bool:
        if not value:
            return False
        v = value.strip().lower()
        if v.startswith("data:video/"):
            return True
        if not (
            v.startswith("http://") or v.startswith("https://") or v.startswith("/")
        ):
            return False
        # Common video extensions / query patterns
        video_exts = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")
        return any(ext in v for ext in video_exts)

    def _normalize_url(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return v
        if v.startswith("data:"):
            return v

        parsed = urlparse(v)
        if parsed.scheme in {"http", "https"}:
            return v

        # Relative path -> join with base url
        return urljoin(self._get_base_url().rstrip("/") + "/", v.lstrip("/"))

    def _extract_video_url(self, task: dict[str, Any]) -> str:
        # 1) Canonical fields
        video_url = task.get("videoUrl")
        if isinstance(video_url, str) and video_url.strip():
            return self._normalize_url(video_url)

        video_urls = task.get("videoUrls")
        if isinstance(video_urls, list) and len(video_urls) > 0:
            for item in video_urls:
                if isinstance(item, str) and item.strip():
                    return self._normalize_url(item)
                if isinstance(item, dict):
                    url = item.get("url") or item.get("videoUrl")
                    if isinstance(url, str) and url.strip():
                        return self._normalize_url(url)

        # 2) Some implementations may put the final asset on generic `url`.
        root_url = task.get("url")
        if isinstance(root_url, str) and self._looks_like_video_url(root_url):
            return self._normalize_url(root_url)

        # 3) Fallback: occasionally the video is stored in `imageUrl`.
        image_url = task.get("imageUrl")
        if isinstance(image_url, str) and self._looks_like_video_url(image_url):
            return self._normalize_url(image_url)

        # 4) Fallback: try `properties` blob.
        props = task.get("properties")
        if isinstance(props, dict):
            for key in ("videoUrl", "video_url", "url", "video"):
                val = props.get(key)
                if isinstance(val, str) and self._looks_like_video_url(val):
                    return self._normalize_url(val)

            # As a last resort, scan for any video-like URL inside properties.
            for val in props.values():
                if isinstance(val, str) and self._looks_like_video_url(val):
                    return self._normalize_url(val)

        raise Exception(
            "No videoUrl found in mjproxy task result (status=%s action=%s)"
            % (task.get("status"), task.get("action"))
        )

    async def _wait_for_video_url(
        self,
        task_id: str,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> str:
        """Wait until task SUCCESS and video URL becomes available.

        Some MJProxy deployments may mark status=SUCCESS before `videoUrl` is populated.
        """

        start_time = asyncio.get_event_loop().time()
        last_task: dict[str, Any] | None = None

        while True:
            task = await self._fetch_task(task_id)
            last_task = task
            status = task.get("status")

            if status == "SUCCESS":
                try:
                    return self._extract_video_url(task)
                except Exception:
                    # Keep polling for URL availability.
                    pass
            elif status in {"FAILURE", "CANCEL"}:
                raise Exception(
                    task.get("failReason") or f"Task {task_id} failed: {status}"
                )
            elif status == "MODAL":
                raise Exception(
                    task.get("failReason")
                    or "Task requires additional confirmation/verification (MODAL)"
                )

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Timeout waiting for mjproxy video url for task {task_id}. "
                    f"Last status={last_task.get('status') if last_task else None}"
                )

            await asyncio.sleep(poll_interval_seconds)

    async def generate(
        self,
        prompt: str,
        model: str,
        resolution: str = "480p",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        input_images: Optional[list[str]] = None,
        camera_fixed: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate video and return a downloadable video URL."""

        try:
            if not input_images or len(input_images) == 0:
                raise ValueError(
                    "MJProxy video generation requires at least 1 input image"
                )

            image = input_images[0]
            end_image = input_images[1] if len(input_images) > 1 else None

            # Default bs1 unless explicitly overridden.
            batch_size = kwargs.get("batch_size")
            if batch_size is None:
                batch_size = kwargs.get("batchSize")
            if batch_size is None:
                batch_size = 1

            task_id = await self._submit_video(
                prompt=prompt or "",
                image=image,
                motion=kwargs.get("motion"),
                end_image=end_image,
                loop=bool(kwargs.get("loop", False)),
                action=str(kwargs.get("action") or "VIDEO"),
                task_id=kwargs.get("task_id") or kwargs.get("taskId"),
                index=kwargs.get("index"),
                video_type=kwargs.get("video_type") or kwargs.get("videoType"),
                batch_size=batch_size,
            )

            # Some MJProxy deployments return only a preview image for VIDEO tasks.
            # In that case, `TaskInfo.buttons` includes a U1 button (video_virtual_upscale)
            # that must be clicked to get the final mp4.
            task = await self._wait_for_task_completion(task_id)
            try:
                return self._extract_video_url(task)
            except Exception:
                u1_custom_id = self._get_u1_custom_id(task)
                if u1_custom_id:
                    up_task_id = await self._submit_action(
                        task_id=task_id,
                        custom_id=u1_custom_id,
                    )
                    return await self._wait_for_video_url(up_task_id)

                # If there is no U1 button, fall back to waiting for the URL to be
                # populated (some implementations fill it after SUCCESS).
                return await self._wait_for_video_url(task_id)

        except Exception as e:
            print("Error generating video with MJProxy:", e)
            traceback.print_exc()
            raise
