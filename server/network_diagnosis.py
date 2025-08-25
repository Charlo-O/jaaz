#!/usr/bin/env python3
"""
网络连接诊断脚本 - 将结果写入文件
"""
import requests
import json
import os
import time
from datetime import datetime

def write_log(message):
    """写入日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("network_diagnosis.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def main():
    write_log("=== 网络连接诊断开始 ===")
    
    # 清空之前的日志
    if os.path.exists("network_diagnosis.log"):
        os.remove("network_diagnosis.log")
    
    # 测试1: 基础连接
    write_log("🔄 测试基础网络连接...")
    try:
        response = requests.get("http://httpbin.org/ip", timeout=10)
        if response.status_code == 200:
            ip_info = response.json()
            write_log(f"✅ 基础网络连接正常 - 状态码: {response.status_code}")
            write_log(f"   当前IP: {ip_info.get('origin', '未知')}")
        else:
            write_log(f"⚠️ 基础网络连接异常 - 状态码: {response.status_code}")
    except Exception as e:
        write_log(f"❌ 基础网络连接失败: {str(e)}")
    
    # 测试2: Google主站
    write_log("🔄 测试Google主站访问...")
    try:
        response = requests.get("https://www.google.com", timeout=15)
        write_log(f"✅ Google主站访问正常 - 状态码: {response.status_code}")
    except Exception as e:
        write_log(f"❌ Google主站访问失败: {str(e)}")
    
    # 测试3: Google API域名
    write_log("🔄 测试Google API域名...")
    try:
        response = requests.get("https://generativelanguage.googleapis.com", timeout=15)
        write_log(f"✅ Google API域名访问正常 - 状态码: {response.status_code}")
    except Exception as e:
        write_log(f"❌ Google API域名访问失败: {str(e)}")
    
    # 测试4: ModelScope API
    write_log("🔄 测试ModelScope API域名...")
    try:
        response = requests.get("https://api-inference.modelscope.cn", timeout=15)
        write_log(f"✅ ModelScope API访问正常 - 状态码: {response.status_code}")
    except Exception as e:
        write_log(f"❌ ModelScope API访问失败: {str(e)}")
    
    # 测试5: Google Gemini API实际调用
    write_log("🔄 测试Google Gemini API调用...")
    api_key = "AIzaSyB_ov8lTrniC3t0zl_fVzV1QgQn9gaxYQY"
    test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    test_data = {
        "contents": [{
            "parts": [{"text": "请回复'连接成功'"}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 50
        }
    }
    
    try:
        response = requests.post(
            test_url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                write_log(f"✅ Google Gemini API调用成功!")
                write_log(f"   AI回复: {content}")
            else:
                write_log(f"⚠️ API响应格式异常")
                write_log(f"   响应: {json.dumps(result, ensure_ascii=False)}")
        else:
            write_log(f"❌ Google Gemini API调用失败 - 状态码: {response.status_code}")
            write_log(f"   错误: {response.text}")
    except Exception as e:
        write_log(f"❌ Google Gemini API调用异常: {str(e)}")
    
    # 测试6: 检查代理设置
    write_log("🔄 检查代理配置...")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
    found_proxy = False
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            write_log(f"   环境变量 {var}: {value}")
            found_proxy = True
    
    if not found_proxy:
        write_log("   未发现代理环境变量")
    
    # 检查settings.json中的代理设置
    try:
        with open("user_data/settings.json", "r", encoding="utf-8") as f:
            settings = json.load(f)
            proxy_setting = settings.get('proxy', '未设置')
            write_log(f"   应用代理设置: {proxy_setting}")
    except Exception as e:
        write_log(f"   无法读取代理设置: {str(e)}")
    
    write_log("=== 网络连接诊断完成 ===")

if __name__ == "__main__":
    main()