#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from services.config_service import FILES_DIR
from tools.utils.image_utils import generate_image_id


class BrowserImageDownloader:
    """使用浏览器自动化下载图片"""
    
    def __init__(self, headless=True, use_proxy=True):
        self.headless = headless
        self.use_proxy = use_proxy
        self.driver = None
        
    def _setup_driver(self):
        """设置Chrome浏览器驱动"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
            
        # 基础设置
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        
        # 用户代理设置
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
        
        # 如果使用系统代理
        if self.use_proxy:
            # Chrome会自动使用系统代理设置
            chrome_options.add_argument('--proxy-auto-detect')
            
        # 下载设置
        prefs = {
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "download.default_directory": FILES_DIR,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            return True
        except Exception as e:
            print(f"❌ 初始化浏览器失败: {e}")
            return False
    
    async def download_image(self, image_url: str, max_retries: int = 3) -> tuple[str, int, int, str]:
        """
        使用浏览器下载图片
        
        Args:
            image_url: 图片URL
            max_retries: 最大重试次数
            
        Returns:
            Tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        
        if not self._setup_driver():
            raise Exception("无法初始化浏览器")
            
        try:
            for attempt in range(max_retries):
                try:
                    print(f'🌐 尝试第 {attempt + 1} 次浏览器下载: {image_url}')
                    
                    # 方法1: 使用JavaScript获取图片数据
                    result = await self._download_via_javascript(image_url)
                    if result:
                        return result
                        
                    # 方法2: 使用右键下载
                    result = await self._download_via_context_menu(image_url)
                    if result:
                        return result
                        
                except Exception as e:
                    print(f'🌐 浏览器下载尝试 {attempt + 1} 失败: {e}')
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 等待后重试
                        
            raise Exception(f"浏览器下载失败，已重试 {max_retries} 次")
            
        finally:
            if self.driver:
                self.driver.quit()
    
    async def _download_via_javascript(self, image_url: str) -> tuple[str, int, int, str] | None:
        """使用JavaScript方式下载图片"""
        try:
            # 创建一个HTML页面来加载图片
            html_content = f"""
            <html>
            <body>
                <img id="target-image" src="{image_url}" crossorigin="anonymous" style="max-width: 100%; height: auto;">
                <canvas id="canvas" style="display: none;"></canvas>
                <script>
                    window.imageLoaded = false;
                    window.imageError = false;
                    window.imageData = null;
                    
                    const img = document.getElementById('target-image');
                    const canvas = document.getElementById('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    img.onload = function() {{
                        try {{
                            canvas.width = img.naturalWidth;
                            canvas.height = img.naturalHeight;
                            ctx.drawImage(img, 0, 0);
                            window.imageData = canvas.toDataURL('image/png');
                            window.imageLoaded = true;
                            window.imageWidth = img.naturalWidth;
                            window.imageHeight = img.naturalHeight;
                        }} catch(e) {{
                            window.imageError = true;
                            window.errorMessage = e.message;
                        }}
                    }};
                    
                    img.onerror = function() {{
                        window.imageError = true;
                        window.errorMessage = 'Image load failed';
                    }};
                </script>
            </body>
            </html>
            """
            
            # 通过data URL加载HTML
            data_url = f"data:text/html;charset=utf-8,{html_content}"
            self.driver.get(data_url)
            
            # 等待图片加载完成
            wait = WebDriverWait(self.driver, 15)
            wait.until(lambda driver: driver.execute_script("return window.imageLoaded || window.imageError"))
            
            # 检查加载结果
            image_loaded = self.driver.execute_script("return window.imageLoaded")
            image_error = self.driver.execute_script("return window.imageError")
            
            if image_error:
                error_msg = self.driver.execute_script("return window.errorMessage")
                print(f"🌐 JavaScript方式加载失败: {error_msg}")
                return None
                
            if not image_loaded:
                print("🌐 JavaScript方式超时")
                return None
            
            # 获取图片数据
            image_data = self.driver.execute_script("return window.imageData")
            width = self.driver.execute_script("return window.imageWidth")
            height = self.driver.execute_script("return window.imageHeight")
            
            if not image_data:
                print("🌐 无法获取图片数据")
                return None
            
            # 解析base64数据
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            
            # 保存文件
            image_id = generate_image_id()
            filename = f"{image_id}.png"
            file_path = os.path.join(FILES_DIR, filename)
            
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f'🌐 JavaScript下载成功: {filename} ({width}x{height})')
            return "image/png", width, height, filename
            
        except Exception as e:
            print(f"🌐 JavaScript下载异常: {e}")
            return None
    
    async def _download_via_context_menu(self, image_url: str) -> tuple[str, int, int, str] | None:
        """使用右键下载方式"""
        try:
            print("🌐 尝试右键下载方式...")
            
            # 直接导航到图片URL
            self.driver.get(image_url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 查找图片元素
            try:
                img_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "img"))
                )
            except:
                print("🌐 未找到图片元素")
                return None
            
            # 获取图片尺寸
            width = self.driver.execute_script("return arguments[0].naturalWidth", img_element)
            height = self.driver.execute_script("return arguments[0].naturalHeight", img_element)
            
            if width == 0 or height == 0:
                print("🌐 图片尺寸无效")
                return None
            
            # 使用JavaScript下载
            image_id = generate_image_id()
            filename = f"{image_id}.png"
            
            download_script = f"""
            var img = arguments[0];
            var canvas = document.createElement('canvas');
            var ctx = canvas.getContext('2d');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0);
            
            canvas.toBlob(function(blob) {{
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = '{filename}';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}, 'image/png');
            """
            
            self.driver.execute_script(download_script, img_element)
            
            # 等待下载完成
            time.sleep(3)
            
            # 检查文件是否下载成功
            file_path = os.path.join(FILES_DIR, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f'🌐 右键下载成功: {filename} ({width}x{height})')
                return "image/png", width, height, filename
            else:
                print("🌐 右键下载失败，文件未生成")
                return None
                
        except Exception as e:
            print(f"🌐 右键下载异常: {e}")
            return None


# 全局下载器实例
_browser_downloader = None

async def download_image_with_browser(image_url: str, headless: bool = True) -> tuple[str, int, int, str]:
    """
    使用浏览器下载图片的便捷函数
    
    Args:
        image_url: 图片URL
        headless: 是否无头模式
        
    Returns:
        Tuple[str, int, int, str]: (mime_type, width, height, filename)
    """
    global _browser_downloader
    
    try:
        downloader = BrowserImageDownloader(headless=headless, use_proxy=True)
        return await downloader.download_image(image_url)
    except Exception as e:
        raise Exception(f"浏览器下载失败: {str(e)}")