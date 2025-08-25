#!/usr/bin/env python3
"""
Test Google Gemini API connection
"""
import requests
import json

def test_google_api():
    api_key = 'AIzaSyB_ov8lTrniC3t0zl_fVzV1QgQn9gaxYQY'
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}'
    
    print('🔄 测试Google Gemini API连接...')
    
    try:
        # 构建请求数据
        data = {
            'contents': [{
                'parts': [{'text': '请回复: 连接测试成功'}]
            }],
            'generationConfig': {
                'temperature': 0.3,
                'maxOutputTokens': 100
            }
        }
        
        # 发送请求
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f'HTTP状态码: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('✅ Google Gemini API连接成功!')
            
            if 'candidates' in result:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print(f'🤖 AI回复: {text}')
                print('💡 视频分析功能可以正常使用')
            else:
                print(f'⚠️ 响应格式异常: {result}')
        else:
            print(f'❌ API调用失败')
            try:
                error_data = response.json()
                print(f'错误信息: {error_data}')
            except:
                print(f'错误文本: {response.text}')
                
    except requests.exceptions.ConnectionError as e:
        print(f'❌ 网络连接错误: {e}')
    except requests.exceptions.Timeout as e:
        print(f'❌ 请求超时: {e}')
    except Exception as e:
        print(f'❌ 其他错误: {e}')

if __name__ == "__main__":
    test_google_api()