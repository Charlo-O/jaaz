from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type


class MusicProviderBase(ABC):
    """Music generation provider base class"""

    # Class attribute: provider registry
    _providers: Dict[str, Type['MusicProviderBase']] = {}

    def __init_subclass__(cls, provider_name: Optional[str] = None, **kwargs: Any):
        """Auto-register provider"""
        super().__init_subclass__(**kwargs)
        if provider_name:
            cls._providers[provider_name] = cls

    @classmethod
    def create_provider(cls, provider_name: str) -> 'MusicProviderBase':
        """Factory method: create provider instance"""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown music provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        return provider_class()

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get all available providers"""
        return list(cls._providers.keys())

    @abstractmethod
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
        Generate music and return result

        Args:
            prompt: Music generation prompt (gpt_description_prompt for inspiration mode)
            model: Model version (chirp-v3-0, chirp-v3-5, chirp-v4, chirp-auk)
            title: Song title (for custom mode)
            tags: Style tags (for custom mode)
            lyrics: Song lyrics (for custom mode)
            make_instrumental: Whether to generate instrumental only
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict containing:
                - task_id: Task ID for polling
                - audio_urls: List of audio URLs (when completed)
                - status: Task status
        """
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status and results

        Args:
            task_id: Task ID to query

        Returns:
            Dict containing task status and results
        """
        pass
