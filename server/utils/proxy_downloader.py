#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import requests
import winreg
from services.config_service import FILES_DIR
from tools.utils.image_utils import generate_image_id, get_image_info_and_save


def get_system_proxy():
    """获取Windows系统代理设置"""
    proxies = {}
    
    try:
        # 读取Windows注册表中的代理设置
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        )
        
        # 检查是否启用代理
        try:
            proxy_enable = winreg.QueryValueEx(key, "ProxyEnable")[0]
            if proxy_enable:
                proxy_server = winreg.QueryValueEx(key, "ProxyServer")[0]
                proxies = {
                    'http': f'http://{proxy_server}',
                    'https': f'http://{proxy_server}'
                }
                print(f'🔧 检测到系统代理: {proxy_server}')
        except FileNotFoundError:
            pass
            
        winreg.CloseKey(key)
        
    except Exception as e:
        print(f'🔧 读取系统代理失败: {e}')
    
    # 检查环境变量代理
    if not proxies:
        if os.environ.get('HTTP_PROXY'):
            proxies['http'] = os.environ.get('HTTP_PROXY')
        if os.environ.get('HTTPS_PROXY'):
            proxies['https'] = os.environ.get('HTTPS_PROXY')
            
        if proxies:
            print(f'🔧 使用环境变量代理: {proxies}')
    
    return proxies


def download_image_with_system_proxy(image_url: str, max_retries: int = 3) -> tuple[str, int, int, str]:
    """
    使用系统代理设置下载图片
    
    Args:
        image_url: 图片URL
        max_retries: 最大重试次数
        
    Returns:
        Tuple[str, int, int, str]: (mime_type, width, height, filename)
    """
    
    # 获取系统代理
    proxies = get_system_proxy()
    
    # 模拟浏览器的完整headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://discord.com/',
        'Sec-Fetch-Dest': 'image',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'same-site',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # 创建session
    session = requests.Session()
    session.headers.update(headers)
    session.proxies.update(proxies)
    
    for attempt in range(max_retries):
        try:
            print(f'🔧 系统代理下载尝试 {attempt + 1}: {image_url}')
            
            response = session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # 获取内容类型
            content_type = response.headers.get('content-type', 'image/png')
            
            # 生成文件名
            image_id = generate_image_id()
            if 'jpeg' in content_type or 'jpg' in content_type:
                filename = f"{image_id}.jpg"
            else:
                filename = f"{image_id}.png"
                
            file_path = os.path.join(FILES_DIR, filename)
            
            # 保存文件
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 获取图片信息
            try:
                mime_type, width, height = get_image_info_and_save(file_path, filename)
                print(f'🔧 系统代理下载成功: {filename} ({width}x{height})')
                return mime_type, width, height, filename
            except Exception as e:
                print(f'🔧 获取图片信息失败: {e}')
                # 返回默认值
                return content_type, 1024, 1024, filename
                
        except Exception as e:
            print(f'🔧 系统代理下载尝试 {attempt + 1} 失败: {e}')
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
    
    raise Exception(f"系统代理下载失败，已重试 {max_retries} 次")