#!/usr/bin/env python3
"""
Simple Google Gemini API test using direct HTTP request
"""
import requests
import json

def test_google_api():
    # Google API配置（从config.toml读取）
    api_key = "AIzaSyB_ov8lTrniC3t0zl_fVzV1QgQn9gaxYQY"
    base_url = "https://generativelanguage.googleapis.com"
    
    print("🔄 测试Google Gemini API连接...")
    print(f"API密钥: {api_key[:8]}...{api_key[-4:]}")
    print(f"API地址: {base_url}")
    print("-" * 50)
    
    # 测试URL
    test_url = f"{base_url}/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    # 测试数据
    test_data = {
        "contents": [{
            "parts": [{"text": "请回复'Google API连接成功'，只回复这一句话"}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 50
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🔄 发送API请求...")
        response = requests.post(
            test_url,
            json=test_data,
            headers=headers,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API请求成功!")
            
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                print(f"🤖 Gemini回复: {content}")
                return True
            else:
                print("❌ 响应格式异常")
                print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return False
        else:
            print(f"❌ API请求失败: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
            except:
                print(f"错误内容: {response.text}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print("❌ 连接超时")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Google Gemini API 连接测试")
    print("=" * 60)
    
    success = test_google_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试结果: Google Gemini API连接成功!")
        print("💡 可以正常使用Gemini视频分析功能")
    else:
        print("❌ 测试结果: Google Gemini API连接失败")
        print("💡 建议检查:")
        print("   1. 网络连接是否正常")
        print("   2. API密钥是否有效")
        print("   3. API配额是否用完")
    print("=" * 60)