import asyncio
import sys
sys.path.append('.')

async def test_gemini_service():
    """测试Gemini服务是否可用"""
    try:
        from services.gemini_video_service import get_gemini_video_service
        
        print("🔄 测试Gemini服务初始化...")
        service = get_gemini_video_service()
        
        if service:
            print("✅ Gemini视频服务初始化成功")
            print(f"   API Base URL: {service.base_url}")
            print(f"   API Key: {service.api_key[:8]}...{service.api_key[-4:]}")
            return True
        else:
            print("❌ Gemini视频服务初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 服务测试异常: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_gemini_service())