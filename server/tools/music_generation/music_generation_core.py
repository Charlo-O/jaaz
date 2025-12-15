"""
Music generation core module
Contains the main orchestration logic for music generation across different providers
"""

import traceback
from typing import Optional, Dict, Any

from ..music_providers.music_base_provider import MusicProviderBase
# Import all providers to ensure automatic registration (don't delete these imports)
from ..music_providers.suno_provider import SunoProvider  # type: ignore


async def generate_music_with_provider(
    prompt: str,
    model: str,
    title: Optional[str] = None,
    tags: Optional[str] = None,
    lyrics: Optional[str] = None,
    make_instrumental: bool = False,
    provider: str = "suno",
    wait_for_completion: bool = True,
    timeout: int = 300,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Universal music generation function supporting different providers

    Args:
        prompt: Music generation prompt (description for inspiration mode)
        model: Model version (e.g., chirp-v4)
        title: Song title (for custom mode)
        tags: Style tags (for custom mode)
        lyrics: Song lyrics (for custom mode)
        make_instrumental: Whether to generate instrumental only
        provider: Provider name (default: suno)
        wait_for_completion: Whether to wait for task completion
        timeout: Maximum wait time in seconds

    Returns:
        Dict containing generation results
    """
    try:
        print(f"🎵 Starting music generation with provider: {provider}, model: {model}")

        # Create provider instance
        provider_instance = MusicProviderBase.create_provider(provider)

        if wait_for_completion and hasattr(provider_instance, 'generate_and_wait'):
            # Generate and wait for completion
            result = await provider_instance.generate_and_wait(
                prompt=prompt,
                model=model,
                title=title,
                tags=tags,
                lyrics=lyrics,
                make_instrumental=make_instrumental,
                timeout=timeout,
                **kwargs
            )
        else:
            # Just submit the task
            result = await provider_instance.generate(
                prompt=prompt,
                model=model,
                title=title,
                tags=tags,
                lyrics=lyrics,
                make_instrumental=make_instrumental,
                **kwargs
            )

        return result

    except Exception as e:
        error_message = str(e)
        print(f"🎵 Error generating music: {error_message}")
        traceback.print_exc()
        raise Exception(f"Music generation failed: {error_message}")


async def get_music_task_status(
    task_id: str,
    provider: str = "suno"
) -> Dict[str, Any]:
    """
    Get music generation task status

    Args:
        task_id: Task ID to query
        provider: Provider name

    Returns:
        Dict containing task status and results
    """
    try:
        provider_instance = MusicProviderBase.create_provider(provider)
        return await provider_instance.get_task_status(task_id)
    except Exception as e:
        print(f"🎵 Error getting music task status: {str(e)}")
        traceback.print_exc()
        raise e
