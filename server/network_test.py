import requests

print("开始网络连接测试...")

# 测试1: 基础网络连接
try:
    response = requests.get("http://httpbin.org/ip", timeout=10)
    print(f"✅ 基础网络连接正常 - 状态码: {response.status_code}")
    print(f"IP地址: {response.json().get('origin', '未知')}")
except Exception as e:
    print(f"❌ 基础网络连接失败: {str(e)}")

# 测试2: Google域名访问
try:
    response = requests.get("https://www.google.com", timeout=10)
    print(f"✅ Google域名访问正常 - 状态码: {response.status_code}")
except Exception as e:
    print(f"❌ Google域名访问失败: {str(e)}")

# 测试3: Google API域名
try:
    response = requests.get("https://generativelanguage.googleapis.com", timeout=10)
    print(f"✅ Google API域名访问正常 - 状态码: {response.status_code}")
except Exception as e:
    print(f"❌ Google API域名访问失败: {str(e)}")

# 测试4: ModelScope API
try:
    response = requests.get("https://api-inference.modelscope.cn", timeout=10)
    print(f"✅ ModelScope API访问正常 - 状态码: {response.status_code}")
except Exception as e:
    print(f"❌ ModelScope API访问失败: {str(e)}")

print("网络连接测试完成。")