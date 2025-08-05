#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import base64
import asyncio
from playwright.async_api import async_playwright
from services.config_service import FILES_DIR
from tools.utils.image_utils import generate_image_id


class PlaywrightImageDownloader:
    """使用Playwright下载图片"""
    
    def __init__(self, headless=True):
        self.headless = headless
        
    async def download_image(self, image_url: str, max_retries: int = 3) -> tuple[str, int, int, str]:
        """
        使用Playwright下载图片
        
        Args:
            image_url: 图片URL
            max_retries: 最大重试次数
            
        Returns:
            Tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        
        async with async_playwright() as p:
            # 启动浏览器 (会自动使用系统代理)
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--proxy-auto-detect',  # 自动检测系统代理
                ]
            )
            
            try:
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                page = await context.new_page()
                
                for attempt in range(max_retries):
                    try:
                        print(f'🎭 Playwright尝试第 {attempt + 1} 次下载: {image_url}')
                        
                        # 创建HTML页面来加载图片
                        html_content = f"""
                        <html>
                        <body>
                            <img id="target-image" src="{image_url}" crossorigin="anonymous">
                            <script>
                                window.imageStatus = 'loading';
                                const img = document.getElementById('target-image');
                                img.onload = () => window.imageStatus = 'loaded';
                                img.onerror = () => window.imageStatus = 'error';
                            </script>
                        </body>
                        </html>
                        """
                        
                        await page.set_content(html_content)
                        
                        # 等待图片加载
                        await page.wait_for_function("window.imageStatus !== 'loading'", timeout=15000)
                        
                        status = await page.evaluate("window.imageStatus")
                        
                        if status == 'error':
                            print(f'🎭 图片加载失败，尝试 {attempt + 1}')
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)
                                continue
                            else:
                                raise Exception("图片加载失败")
                        
                        # 获取图片信息
                        img_info = await page.evaluate("""
                            const img = document.getElementById('target-image');
                            const canvas = document.createElement('canvas');
                            const ctx = canvas.getContext('2d');
                            
                            canvas.width = img.naturalWidth;
                            canvas.height = img.naturalHeight;
                            ctx.drawImage(img, 0, 0);
                            
                            return {
                                width: img.naturalWidth,
                                height: img.naturalHeight,
                                dataUrl: canvas.toDataURL('image/png')
                            };
                        """)
                        
                        if not img_info['dataUrl']:
                            raise Exception("无法获取图片数据")
                        
                        # 解析并保存图片
                        header, encoded = img_info['dataUrl'].split(',', 1)
                        image_bytes = base64.b64decode(encoded)
                        
                        image_id = generate_image_id()
                        filename = f"{image_id}.png"
                        file_path = os.path.join(FILES_DIR, filename)
                        
                        with open(file_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        print(f'🎭 Playwright下载成功: {filename} ({img_info["width"]}x{img_info["height"]})')
                        return "image/png", img_info["width"], img_info["height"], filename
                        
                    except Exception as e:
                        print(f'🎭 Playwright尝试 {attempt + 1} 失败: {e}')
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                
                raise Exception(f"Playwright下载失败，已重试 {max_retries} 次")
                
            finally:
                await browser.close()


async def download_image_with_playwright(image_url: str, headless: bool = True) -> tuple[str, int, int, str]:
    """
    使用Playwright下载图片的便捷函数
    
    Args:
        image_url: 图片URL
        headless: 是否无头模式
        
    Returns:
        Tuple[str, int, int, str]: (mime_type, width, height, filename)
    """
    downloader = PlaywrightImageDownloader(headless=headless)
    return await downloader.download_image(image_url)