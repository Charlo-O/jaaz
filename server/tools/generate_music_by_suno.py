"""
Suno music generation tool
Supports both inspiration mode and custom mode
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig

from tools.music_generation.music_generation_core import generate_music_with_provider
from common import DEFAULT_PORT


class GenerateMusicBySunoInputSchema(BaseModel):
    prompt: str = Field(
        description="Required. The prompt for music generation. Describe the style, mood, theme, and language of the song you want to create. Example: 'A cheerful pop song in Chinese about summer vacation'"
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional. The title of the song. If provided, switches to custom mode."
    )
    tags: Optional[str] = Field(
        default=None,
        description="Optional. Style tags for the song, e.g., 'pop rock emotional'. If provided, switches to custom mode."
    )
    lyrics: Optional[str] = Field(
        default=None,
        description="Optional. Custom lyrics for the song. Use [Verse], [Chorus], [Bridge] markers. If provided, switches to custom mode. Empty string creates instrumental."
    )
    make_instrumental: bool = Field(
        default=False,
        description="Optional. Set to true to generate instrumental music without vocals. Only used in inspiration mode."
    )
    model: str = Field(
        default="chirp-v4",
        description="Optional. Model version to use: chirp-v3-0, chirp-v3-5, chirp-v4, chirp-auk (v4.5). Default is chirp-v4."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


@tool(args_schema=GenerateMusicBySunoInputSchema)
async def generate_music_by_suno(
    prompt: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    title: Optional[str] = None,
    tags: Optional[str] = None,
    lyrics: Optional[str] = None,
    make_instrumental: bool = False,
    model: str = "chirp-v4",
) -> str:
    """
    Generate music using Suno AI. Supports two modes:
    
    1. Inspiration Mode: Just provide a prompt describing the song you want. Suno will automatically generate title, lyrics, style, and music.
    
    2. Custom Mode: Provide title, tags (style), and/or lyrics for more control over the output.
    
    The tool will wait for the music generation to complete and return the audio URLs.
    """
    ctx = config.get('configurable', {})
    canvas_id = ctx.get('canvas_id', '')
    session_id = ctx.get('session_id', '')
    
    print(f'🎵 Suno Music Generation tool_call_id: {tool_call_id}')
    print(f'🎵 canvas_id: {canvas_id}, session_id: {session_id}')

    try:
        result = await generate_music_with_provider(
            prompt=prompt,
            model=model,
            title=title,
            tags=tags,
            lyrics=lyrics,
            make_instrumental=make_instrumental,
            provider="suno",
            wait_for_completion=True,
            timeout=300,
        )

        # Format the response
        songs = result.get("songs", [])
        audio_urls = result.get("audio_urls", [])
        
        if not songs and not audio_urls:
            return f"Music generation completed but no songs were returned. Task ID: {result.get('task_id', 'unknown')}"

        # Build response message
        response_parts = ["🎵 Music generated successfully!\n"]
        
        for i, song in enumerate(songs, 1):
            song_title = song.get("title", f"Song {i}")
            audio_url = song.get("audio_url", "")
            video_url = song.get("video_url", "")
            duration = song.get("duration", 0)
            song_tags = song.get("tags", "")
            
            response_parts.append(f"\n**{song_title}**")
            if song_tags:
                response_parts.append(f"- Style: {song_tags}")
            if duration:
                response_parts.append(f"- Duration: {duration:.1f}s")
            if audio_url:
                response_parts.append(f"- Audio: {audio_url}")
            if video_url:
                response_parts.append(f"- Video: {video_url}")

        return "\n".join(response_parts)

    except Exception as e:
        error_msg = str(e)
        print(f"🎵 Error in generate_music_by_suno: {error_msg}")
        return f"Failed to generate music: {error_msg}"
