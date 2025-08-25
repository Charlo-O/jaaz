#!/usr/bin/env python3
"""
Test script to check ModelScope OpenAI-compatible API connection
"""
import requests
import json
import sys
from typing import Dict, Any

def test_modelscope_api():
    """Test connection to ModelScope API"""
    
    url = "https://api-inference.modelscope.cn/v1/chat/completions"
    api_key = "ms-50dad399-1268-4772-b5ed-18ca2560462f"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "测试API连接，请回复'连接成功'"}
        ],
        "max_tokens": 50,
        "temperature": 0.3
    }
    
    print("🔄 Testing ModelScope API connection...")
    print(f"URL: {url}")
    print(f"Model: {payload['model']}")
    print("-" * 50)
    
    try:
        # Test with timeout
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API连接成功!")
            print("Response:", json.dumps(result, indent=2, ensure_ascii=False))
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"🤖 AI回复: {content}")
            
            return True
        else:
            print(f"❌ API请求失败")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print("❌ 连接超时 - 无法连接到ModelScope服务器")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {str(e)}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    except json.JSONDecodeError:
        print(f"❌ JSON解析错误 - 响应内容: {response.text}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return False

def test_basic_connectivity():
    """Test basic internet connectivity"""
    print("🔄 Testing basic internet connectivity...")
    
    try:
        # Test with a reliable service
        response = requests.get("https://httpbin.org/ip", timeout=10)
        if response.status_code == 200:
            print("✅ 基本网络连接正常")
            print(f"当前IP: {response.json().get('origin', 'unknown')}")
            return True
        else:
            print("❌ 基本网络连接异常")
            return False
    except Exception as e:
        print(f"❌ 网络连接测试失败: {str(e)}")
        return False

def test_modelscope_domain():
    """Test if ModelScope domain is accessible"""
    print("🔄 Testing ModelScope domain accessibility...")
    
    try:
        # Test basic domain access
        response = requests.get("https://modelscope.cn", timeout=15)
        if response.status_code == 200:
            print("✅ ModelScope网站可访问")
            return True
        else:
            print(f"❌ ModelScope网站访问异常 - 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ModelScope网站访问失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ModelScope API 连接测试")
    print("=" * 60)
    
    # Step 1: Basic connectivity
    if not test_basic_connectivity():
        print("\n❌ 基本网络连接失败，请检查网络设置")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    
    # Step 2: ModelScope domain
    if not test_modelscope_domain():
        print("\n⚠️ ModelScope网站访问有问题，可能影响API调用")
    
    print("\n" + "-" * 60)
    
    # Step 3: API test
    success = test_modelscope_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试结果: ModelScope API连接成功!")
        print("💡 LangGraph的连接问题可能是其他原因导致的")
    else:
        print("❌ 测试结果: ModelScope API连接失败")
        print("💡 建议:")
        print("   1. 检查API密钥是否有效")
        print("   2. 确认网络环境是否有防火墙/代理限制")
        print("   3. 考虑切换到其他API提供商（如Google Gemini）")
    print("=" * 60)