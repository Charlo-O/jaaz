#!/usr/bin/env python3
"""
Test script to check Google Gemini API connection
"""
import asyncio
import sys
from services.config_service import config_service
from services.gemini_video_service import get_gemini_video_service
import httpx

async def test_google_basic_connectivity():
    """Test basic connectivity to Google services"""
    print("🔄 Testing basic connectivity to Google services...")
    
    try:
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get("https://generativelanguage.googleapis.com")
            print(f"✅ Google API域名可访问 - 状态码: {response.status_code}")
            return True
    except Exception as e:
        print(f"❌ Google API域名访问失败: {str(e)}")
        return False

async def test_google_api_key():
    """Test Google API key configuration"""
    print("🔄 Testing Google API key configuration...")
    
    try:
        config = config_service.app_config.get('google', {})
        api_key = config.get('api_key', '')
        
        if not api_key:
            print("❌ Google API密钥未配置")
            return False
        
        # Mask API key for display
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"✅ Google API密钥已配置: {masked_key}")
        return True
        
    except Exception as e:
        print(f"❌ 配置读取失败: {str(e)}")
        return False

async def test_gemini_service_init():
    """Test Gemini service initialization"""
    print("🔄 Testing Gemini service initialization...")
    
    try:
        gemini_service = get_gemini_video_service()
        if gemini_service:
            print("✅ Gemini视频服务初始化成功")
            return True
        else:
            print("❌ Gemini视频服务初始化失败")
            return False
    except Exception as e:
        print(f"❌ Gemini服务初始化异常: {str(e)}")
        return False

async def test_gemini_api_simple():
    """Test simple Gemini API call"""
    print("🔄 Testing simple Gemini API call...")
    
    try:
        config = config_service.app_config.get('google', {})
        api_key = config.get('api_key', '')
        
        if not api_key:
            print("❌ API密钥未配置，跳过API测试")
            return False
        
        # Simple text generation test
        test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        test_data = {
            "contents": [{
                "parts": [{"text": "请回复'API连接成功'"}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 50
            }
        }
        
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                test_url,
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    print(f"✅ Google Gemini API连接成功!")
                    print(f"🤖 AI回复: {content}")
                    return True
                else:
                    print("❌ API响应格式异常")
                    return False
            else:
                error_text = response.text
                print(f"❌ API调用失败: {response.status_code}")
                print(f"错误详情: {error_text}")
                return False
                
    except Exception as e:
        print(f"❌ API测试异常: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Google Gemini API 连接测试")
    print("=" * 60)
    
    # Test 1: Basic connectivity
    connectivity_ok = await test_google_basic_connectivity()
    
    print("\n" + "-" * 60)
    
    # Test 2: API key configuration
    config_ok = await test_google_api_key()
    
    print("\n" + "-" * 60)
    
    # Test 3: Service initialization
    service_ok = await test_gemini_service_init()
    
    print("\n" + "-" * 60)
    
    # Test 4: Simple API call (only if config is OK)
    api_ok = False
    if config_ok:
        api_ok = await test_gemini_api_simple()
    else:
        print("⚠️ 跳过API调用测试（配置问题）")
    
    print("\n" + "=" * 60)
    
    # Summary
    if connectivity_ok and config_ok and service_ok and api_ok:
        print("🎉 测试结果: Google Gemini API连接完全正常!")
        print("💡 视频分析功能可以正常使用")
        sys.exit(0)
    else:
        print("❌ 测试结果: Google Gemini API连接存在问题")
        print("\n💡 问题排查建议:")
        
        if not connectivity_ok:
            print("   1. 检查网络连接和防火墙设置")
        if not config_ok:
            print("   2. 在设置中配置有效的Google AI API密钥")
        if not service_ok:
            print("   3. 检查服务配置文件和依赖包")
        if config_ok and not api_ok:
            print("   4. 检查API密钥权限和配额")
            
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())