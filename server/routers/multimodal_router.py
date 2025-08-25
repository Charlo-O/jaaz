from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.multimodal_ai_service import multimodal_ai_service

router = APIRouter(prefix="/api")

class MediaAnalysisRequest(BaseModel):
    media_content: str  # Base64 data URL
    media_type: str     # "image" or "video"
    prompt: Optional[str] = ""
    provider: Optional[str] = "openai"
    model: Optional[str] = ""

class MediaAnalysisResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    media_type: Optional[str] = None

@router.post("/analyze_media", response_model=MediaAnalysisResponse)
async def analyze_media(request: MediaAnalysisRequest):
    """
    Analyze image or video content with AI models
    """
    try:
        result = await multimodal_ai_service.analyze_media(
            media_content=request.media_content,
            media_type=request.media_type,
            prompt=request.prompt,
            preferred_provider=request.provider,
            preferred_model=request.model
        )
        
        return MediaAnalysisResponse(
            success=result.get('success', False),
            result=result.get('result'),
            error=result.get('error'),
            provider=result.get('provider'),
            model=result.get('model'),
            media_type=result.get('media_type')
        )
        
    except Exception as e:
        return MediaAnalysisResponse(
            success=False,
            error=str(e)
        )

@router.get("/supported_models")
async def get_supported_models():
    """
    Get list of supported multimodal AI models and their capabilities
    """
    try:
        models = multimodal_ai_service.get_supported_models()
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }