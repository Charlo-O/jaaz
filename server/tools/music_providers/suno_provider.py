"""
Suno music generation provider implementation
API Documentation: https://s.apifox.cn/a7668f05-b561-4ab3-9f93-9798942d810c/llms.txt
"""

import asyncio
import traceback
from typing import Optional, Dict, Any, List

from .music_base_provider import MusicProviderBase
from utils.http_client import HttpClient
from services.config_service import config_service


class SunoProvider(MusicProviderBase, provider_name="suno"):
    """Suno music generation provider implementation"""

    def __init__(self):
        config = config_service.app_config.get('suno', {})
        self.api_key = config.get("api_key", "")
        # Remove trailing /v1 or /v1/ from URL since Suno endpoints don't use /v1 prefix
        base_url = config.get("url", "https://api.no1api.com").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("Suno API key is not configured")

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_payload(
        self,
        prompt: str,
        model: str = "chirp-v4",
        title: Optional[str] = None,
        tags: Optional[str] = None,
        lyrics: Optional[str] = None,
        make_instrumental: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Build request payload for Suno API
        
        Supports two modes:
        1. Inspiration mode: Only gpt_description_prompt, make_instrumental, mv
        2. Custom mode: prompt (lyrics), title, tags, mv
        """
        payload: Dict[str, Any] = {
            "mv": model,  # Model version
        }

        # Determine mode based on parameters
        if lyrics or title or tags:
            # Custom mode - use provided lyrics, title, tags
            payload["prompt"] = lyrics or ""  # Empty string = instrumental
            if title:
                payload["title"] = title
            if tags:
                payload["tags"] = tags
        else:
            # Inspiration mode - use gpt_description_prompt
            payload["gpt_description_prompt"] = prompt
            payload["make_instrumental"] = make_instrumental

        # Add optional callback hook if provided
        if kwargs.get("notify_hook"):
            payload["notify_hook"] = kwargs["notify_hook"]

        return payload

    async def generate(
        self,
        prompt: str,
        model: str = "chirp-v4",
        title: Optional[str] = None,
        tags: Optional[str] = None,
        lyrics: Optional[str] = None,
        make_instrumental: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate music using Suno API

        Returns:
            Dict containing task_id and initial status
        """
        try:
            api_url = f"{self.base_url}/suno/submit/music"
            headers = self._build_headers()

            payload = self._build_request_payload(
                prompt=prompt,
                model=model,
                title=title,
                tags=tags,
                lyrics=lyrics,
                make_instrumental=make_instrumental,
                **kwargs
            )

            print(f"🎵 Starting Suno music generation with model: {model}")
            print(f"🎵 Payload: {payload}")

            async with HttpClient.create_aiohttp() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            error_message = error_data.get("message", f"HTTP {response.status}")
                        except Exception:
                            error_text = await response.text()
                            error_message = f"HTTP {response.status}: {error_text[:200]}"
                        raise Exception(f"Suno task creation failed: {error_message}")

                    result = await response.json()
                    
                    if result.get("code") != "success":
                        raise Exception(f"Suno API error: {result.get('message', 'Unknown error')}")
                    
                    task_id = result.get("data")
                    
                    if not task_id:
                        raise Exception("No task_id returned from Suno API")

                    print(f"🎵 Suno music generation task created, task_id: {task_id}")

                    return {
                        "task_id": task_id,
                        "status": "submitted",
                        "audio_urls": [],
                    }

        except Exception as e:
            print(f"🎵 Error generating music with Suno: {str(e)}")
            traceback.print_exc()
            raise e

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status and results from Suno API

        Returns:
            Dict containing task status and results
        """
        try:
            api_url = f"{self.base_url}/suno/fetch/{task_id}"
            headers = self._build_headers()

            async with HttpClient.create_aiohttp() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            error_message = error_data.get("message", f"HTTP {response.status}")
                        except Exception:
                            error_text = await response.text()
                            error_message = f"HTTP {response.status}: {error_text[:200]}"
                        raise Exception(f"Suno task query failed: {error_message}")

                    result = await response.json()
                    
                    if result.get("code") != "success":
                        raise Exception(f"Suno API error: {result.get('message', 'Unknown error')}")

                    raw_data = result.get("data")
                    data: Dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
                    status = data.get("status", "unknown")
                    progress = data.get("progress", "0%")
                    fail_reason = data.get("fail_reason", "")

                    # Extract audio URLs from completed songs
                    audio_urls: List[str] = []
                    video_urls: List[str] = []
                    songs: List[Dict[str, Any]] = []

                    songs_data = data.get("data")
                    if not isinstance(songs_data, list):
                        if songs_data is not None:
                            print(
                                f"🎵 Unexpected Suno response format: data.data is {type(songs_data)}"
                            )
                        songs_data = []

                    for song in songs_data:
                        if not isinstance(song, dict):
                            continue
                        if song.get("status") == "complete":
                            audio_url = song.get("audio_url") or ""
                            video_url = song.get("video_url") or ""
                            if audio_url:
                                audio_urls.append(audio_url)
                            if video_url:
                                video_urls.append(video_url)

                            metadata = song.get("metadata") or {}
                            if not isinstance(metadata, dict):
                                metadata = {}

                            songs.append({
                                "id": song.get("id", ""),
                                "title": song.get("title", ""),
                                "audio_url": audio_url,
                                "video_url": video_url,
                                "image_url": song.get("image_url", ""),
                                "duration": metadata.get("duration", 0),
                                "tags": metadata.get("tags", ""),
                            })

                    return {
                        "task_id": task_id,
                        "status": status,
                        "progress": progress,
                        "fail_reason": fail_reason,
                        "audio_urls": audio_urls,
                        "video_urls": video_urls,
                        "songs": songs,
                    }

        except Exception as e:
            print(f"🎵 Error querying Suno task status: {str(e)}")
            traceback.print_exc()
            raise e

    async def generate_and_wait(
        self,
        prompt: str,
        model: str = "chirp-v4",
        title: Optional[str] = None,
        tags: Optional[str] = None,
        lyrics: Optional[str] = None,
        make_instrumental: bool = False,
        timeout: int = 300,
        poll_interval: int = 5,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate music and wait for completion

        Args:
            timeout: Maximum wait time in seconds (default 300s = 5 minutes)
            poll_interval: Polling interval in seconds (default 5s)

        Returns:
            Dict containing completed task results
        """
        # Start generation
        result = await self.generate(
            prompt=prompt,
            model=model,
            title=title,
            tags=tags,
            lyrics=lyrics,
            make_instrumental=make_instrumental,
            **kwargs
        )

        task_id = result["task_id"]
        elapsed = 0

        # Poll for completion
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            status_result = await self.get_task_status(task_id)
            status = status_result.get("status", "")
            progress = status_result.get("progress", "0%")

            print(f"🎵 Polling Suno task {task_id}, status: {status}, progress: {progress}")

            if status == "SUCCESS":
                print(f"🎵 Suno music generation completed!")
                return status_result
            elif status in ("FAILED", "failed"):
                fail_reason = status_result.get("fail_reason", "Unknown error")
                raise Exception(f"Suno music generation failed: {fail_reason}")

        raise Exception(f"Suno music generation timed out after {timeout} seconds")
