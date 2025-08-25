#!/usr/bin/env python3
"""
Simple Google API test
"""
import asyncio
import httpx
import json

async def test_google_connection():
    """Simple test"""
    print("🔄 测试Google API基础连接...")
    
    try:
        # Test 1: Basic connectivity
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get("https://generativelanguage.googleapis.com")
            print(f"✅ Google API域名连接成功 - 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ Google API域名连接失败: {str(e)}")
        return False
    
    # Test 2: Check config file
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            google_config = config.get('google', {})
            api_key = google_config.get('api_key', '')
            
            if api_key:
                masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
                print(f"✅ Google API密钥已配置: {masked_key}")
                
                # Test 3: Simple API call
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                
                test_data = {
                    "contents": [{
                        "parts": [{"text": "请回复'连接成功'"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 50
                    }
                }
                
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
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
                            print(f"🤖 回复: {content}")
                            return True
                        else:
                            print("❌ API响应格式异常")
                            print(f"响应内容: {result}")
                            return False
                    else:
                        print(f"❌ API调用失败: {response.status_code}")
                        print(f"错误: {response.text}")
                        return False
            else:
                print("❌ Google API密钥未配置")
                return False
                
    except FileNotFoundError:
        print("❌ 配置文件config.json不存在")
        return False
    except Exception as e:
        print(f"❌ 配置读取失败: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_google_connection())