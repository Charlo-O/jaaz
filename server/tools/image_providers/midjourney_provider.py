import os
import traceback
import asyncio
from typing import Optional, Any, Tuple
from .image_base_provider import ImageProviderBase
from ..utils.image_utils import get_image_info_and_save, generate_image_id
from services.config_service import FILES_DIR
from utils.http_client import HttpClient
from services.config_service import config_service
import time
import aiohttp
import json


class MidjourneyProvider(ImageProviderBase):
    """Midjourney Proxy API image generation provider implementation"""

    def _build_url(self, endpoint: str) -> str:
        """Build request URL for Midjourney Proxy API"""
        config = config_service.app_config.get('midjourney', {})
        api_url = str(config.get("url", "")).rstrip("/")
        
        if not api_url:
            raise ValueError("Midjourney API URL is not configured")
        
        return f"{api_url}/mj/{endpoint}"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers"""
        config = config_service.app_config.get('midjourney', {})
        api_key = config.get("api_key", "")
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add API key to headers if configured
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        return headers

    async def _submit_imagine_task(self, prompt: str, base64_images: Optional[list[str]] = None) -> str:
        """
        Submit imagine task to Midjourney API
        
        Returns:
            str: Task ID
        """
        url = self._build_url("submit/imagine")
        headers = self._build_headers()
        
        data = {
            "prompt": prompt,
            "base64Array": base64_images or [],
            "notifyHook": "",
            "state": ""
        }
        
        async with HttpClient.create_aiohttp() as session:
            print(f'🎨 Midjourney API request: {url}, prompt: {prompt[:100]}...')
            async with session.post(url, headers=headers, json=data) as response:
                json_data = await response.json()
                print(f'🎨 Midjourney API response: {json_data}')
                
                if json_data.get("code") == 1:
                    return json_data.get("result")
                else:
                    raise ValueError(f"Midjourney API error: {json_data.get('description', 'Unknown error')}")

    async def _submit_upscale_task(self, task_id: str, index: int) -> str:
        """
        Submit upscale task to Midjourney API
        
        Args:
            task_id: Original task ID
            index: Image index (1-4)
            
        Returns:
            str: Upscale task ID
        """
        url = self._build_url("submit/change")
        headers = self._build_headers()
        
        data = {
            "action": "UPSCALE",
            "index": index,
            "taskId": task_id,
            "notifyHook": "",
            "state": ""
        }
        
        async with HttpClient.create_aiohttp() as session:
            print(f'🎨 Midjourney upscale request: {url}, task_id: {task_id}, index: {index}')
            async with session.post(url, headers=headers, json=data) as response:
                json_data = await response.json()
                print(f'🎨 Midjourney upscale response: {json_data}')
                
                if json_data.get("code") == 1:
                    return json_data.get("result")
                else:
                    raise ValueError(f"Midjourney upscale API error: {json_data.get('description', 'Unknown error')}")

    async def _wait_for_task_completion(self, task_id: str, max_wait_time: int = 300) -> dict[str, Any]:
        """
        Wait for task completion and return task details
        
        Args:
            task_id: Task ID to wait for
            max_wait_time: Maximum wait time in seconds
            
        Returns:
            dict: Task details
        """
        url = self._build_url(f"task/{task_id}/fetch")
        headers = self._build_headers()
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            async with HttpClient.create_aiohttp() as session:
                async with session.get(url, headers=headers) as response:
                    task_data = await response.json()
                    
                    status = task_data.get("status")
                    progress = task_data.get("progress", "0%")
                    
                    print(f'🎨 Midjourney task {task_id} status: {status}, progress: {progress}')
                    
                    if status == "SUCCESS":
                        return task_data
                    elif status == "FAILURE":
                        fail_reason = task_data.get('failReason', 'Unknown error')
                        # 检查是否是网络/SSL相关错误
                        if any(keyword in fail_reason.lower() for keyword in 
                               ['ssl', 'handshake', 'remote host terminated', 'connection', 'timeout']):
                            print(f'🔍 检测到网络连接问题，建议检查代理配置')
                            print(f'💡 Midjourney代理服务器也无法访问Discord API')
                            print(f'💡 建议为Midjourney代理服务器配置网络代理')
                        raise ValueError(f"Midjourney task failed: {fail_reason}")
                    
                    # Wait before next check
                    await asyncio.sleep(5)
        
        raise TimeoutError(f"Midjourney task {task_id} did not complete within {max_wait_time} seconds")

    async def _diagnose_network_issue(self, image_url: str) -> str:
        """诊断网络连接问题"""
        import socket
        from urllib.parse import urlparse
        
        diagnostic_info = []
        parsed_url = urlparse(image_url)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        
        # 1. DNS解析测试
        try:
            ip_address = socket.gethostbyname(host)
            diagnostic_info.append(f"✅ DNS解析成功: {host} -> {ip_address}")
        except socket.gaierror as e:
            diagnostic_info.append(f"❌ DNS解析失败: {host} - {str(e)}")
            return "\n".join(diagnostic_info)
        
        # 2. TCP连接测试
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                diagnostic_info.append(f"✅ TCP连接成功: {host}:{port}")
            else:
                diagnostic_info.append(f"❌ TCP连接失败: {host}:{port} - 错误码: {result}")
        except Exception as e:
            diagnostic_info.append(f"❌ TCP连接测试异常: {str(e)}")
        
        # 3. 简单HTTP测试
        try:
            import requests
            # 尝试使用系统代理
            proxies = {}
            if os.environ.get('HTTP_PROXY'):
                proxies['http'] = os.environ.get('HTTP_PROXY')
            if os.environ.get('HTTPS_PROXY'):
                proxies['https'] = os.environ.get('HTTPS_PROXY')
            
            response = requests.head(image_url, timeout=10, allow_redirects=True, proxies=proxies)
            diagnostic_info.append(f"✅ HTTP响应: {response.status_code} - {response.reason}")
            diagnostic_info.append(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
            diagnostic_info.append(f"   Content-Length: {response.headers.get('Content-Length', 'N/A')}")
            if proxies:
                diagnostic_info.append(f"   使用代理: {proxies}")
        except requests.exceptions.ConnectTimeout:
            diagnostic_info.append(f"❌ HTTP连接超时")
        except requests.exceptions.ConnectionError as e:
            diagnostic_info.append(f"❌ HTTP连接错误: {str(e)}")
        except Exception as e:
            diagnostic_info.append(f"❌ HTTP测试异常: {str(e)}")
        
        # 4. 代理配置建议
        if not os.environ.get('HTTP_PROXY') and not os.environ.get('HTTPS_PROXY'):
            diagnostic_info.append(f"\n💡 建议配置代理:")
            diagnostic_info.append(f"   set HTTPS_PROXY=http://127.0.0.1:7897")
            diagnostic_info.append(f"   set HTTP_PROXY=http://127.0.0.1:7897")
        
        return "\n".join(diagnostic_info)

    async def _process_task_result(self, task_data: dict[str, Any]) -> Tuple[str, int, int, str]:
        """
        Process task result and save image
        
        Args:
            task_data: Task data from Midjourney API
            
        Returns:
            Tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        image_url = task_data.get("imageUrl")
        if not image_url:
            raise ValueError("No image URL in task result")
        
        # Download and save the image with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=60, connect=30)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': 'https://discord.com/',
                    'Sec-Fetch-Dest': 'image',
                    'Sec-Fetch-Mode': 'no-cors',
                    'Sec-Fetch-Site': 'same-site',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
                
                # Use trust_env=True to respect proxy settings
                async with HttpClient.create_aiohttp(timeout=timeout, trust_env=True) as session:
                    async with session.get(image_url, headers=headers) as response:
                        if response.status != 200:
                            raise ValueError(f"Failed to download image: HTTP {response.status}")
                        
                        image_data = await response.read()
                        
                        # Generate unique filename
                        image_id = generate_image_id()
                        filename = f"{image_id}.png"
                        
                        # Save image to files directory
                        file_path = os.path.join(FILES_DIR, filename)
                        with open(file_path, "wb") as f:
                            f.write(image_data)
                        
                        # Get image info
                        mime_type, width, height = get_image_info_and_save(file_path, filename)
                        
                        print(f'🎨 Midjourney image saved: {filename} ({width}x{height})')
                        
                        return mime_type, width, height, filename
                        
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f'🎨 Download attempt {attempt + 1} failed: {str(e)}, retrying in {retry_delay} seconds...')
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # 尝试多种下载方式
                    print(f'🎨 网络下载失败，尝试其他下载方式...')
                    
                    # 方式1: 使用系统代理
                    try:
                        from utils.proxy_downloader import download_image_with_system_proxy
                        return download_image_with_system_proxy(image_url)
                    except Exception as proxy_error:
                        print(f'🔧 系统代理下载失败: {proxy_error}')
                    
                    # 方式2: 使用浏览器自动化 (Selenium)
                    try:
                        from utils.browser_downloader import download_image_with_browser
                        return await download_image_with_browser(image_url, headless=True)
                    except Exception as browser_error:
                        print(f'🌐 浏览器下载失败: {browser_error}')
                    
                    # 方式3: 使用Playwright (如果安装了的话)
                    try:
                        from utils.playwright_downloader import download_image_with_playwright
                        return await download_image_with_playwright(image_url, headless=True)
                    except Exception as playwright_error:
                        print(f'🎭 Playwright下载失败: {playwright_error}')
                    
                    # 所有方式都失败，进行诊断并返回URL
                    print(f'🎨 所有下载方式都失败，进行网络诊断...')
                    
                    # 运行网络诊断
                    diagnostic_result = await self._diagnose_network_issue(image_url)
                    print(f'🔍 网络诊断结果:\n{diagnostic_result}')
                    
                    # 返回图片URL给前端处理
                    print(f'🎨 返回图片URL供前端显示: {image_url}')
                    
                    # 生成包含URL的JSON文件
                    image_id = generate_image_id()
                    json_filename = f"{image_id}_info.json"
                    json_file_path = os.path.join(FILES_DIR, json_filename)
                    
                    # 保存图片信息到JSON文件
                    image_info = {
                        "type": "remote_image",
                        "imageUrl": image_url,
                        "width": 1024,
                        "height": 1024,
                        "downloadError": str(e),
                        "attempts": {
                            "direct": "failed",
                            "system_proxy": "failed", 
                            "selenium": "failed",
                            "playwright": "failed"
                        },
                        "diagnostic": diagnostic_result,
                        "timestamp": time.time(),
                        "suggestion": "请在前端使用浏览器直接显示此图片URL"
                    }
                    
                    with open(json_file_path, "w", encoding="utf-8") as f:
                        json.dump(image_info, f, indent=2, ensure_ascii=False)
                    
                    # 返回JSON文件信息，前端可以解析并直接显示图片URL
                    return "application/json", 1024, 1024, json_filename
            except Exception as e:
                raise ValueError(f"Unexpected error downloading image: {str(e)}")

    async def generate_with_upscale(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> list[Tuple[str, int, int, str]]:
        """
        Generate image using Midjourney and upscale all 4 images with 5s intervals
        
        Args:
            prompt: Image generation prompt
            model: Model name (not used for Midjourney)
            aspect_ratio: Image aspect ratio
            input_images: Optional input images for reference
            metadata: Optional metadata to be saved
            **kwargs: Additional parameters
            
        Returns:
            list[Tuple[str, int, int, str]]: List of 4 upscaled images (mime_type, width, height, filename)
        """
        try:
            # Convert aspect ratio to Midjourney format
            aspect_ratio_map = {
                "1:1": " --ar 1:1",
                "16:9": " --ar 16:9", 
                "9:16": " --ar 9:16",
                "4:3": " --ar 4:3",
                "3:4": " --ar 3:4"
            }
            
            # Add aspect ratio to prompt if specified
            enhanced_prompt = prompt
            if aspect_ratio in aspect_ratio_map:
                enhanced_prompt += aspect_ratio_map[aspect_ratio]
            
            # Convert input images to base64 if provided
            base64_images = None
            if input_images:
                base64_images = []
                for image_path in input_images:
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            import base64
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                            # Determine image format
                            if image_path.lower().endswith('.png'):
                                base64_images.append(f"data:image/png;base64,{image_data}")
                            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                                base64_images.append(f"data:image/jpeg;base64,{image_data}")
                            else:
                                base64_images.append(f"data:image/png;base64,{image_data}")
            
            # Step 1: Submit imagine task to generate 4-grid image
            print(f'🎨 Step 1: Generating 4-grid image with prompt: {enhanced_prompt[:100]}...')
            original_task_id = await self._submit_imagine_task(enhanced_prompt, base64_images)
            
            # Wait for 4-grid completion
            print(f'🎨 Waiting for 4-grid task completion: {original_task_id}')
            await self._wait_for_task_completion(original_task_id)
            print(f'🎨 4-grid image generation completed!')
            
            # Step 2: Submit all 4 upscale requests with 5s intervals (don't wait for completion)
            upscale_task_ids = []
            for i in range(1, 5):
                print(f'🎨 Step 2.{i}: Submitting upscale request for image {i}/4')
                
                # Submit upscale task (don't wait for completion)
                upscale_task_id = await self._submit_upscale_task(original_task_id, i)
                upscale_task_ids.append(upscale_task_id)
                
                print(f'🎨 Upscale task {i}/4 submitted with ID: {upscale_task_id}')
                
                # Wait 5 seconds before next upscale request (except for the last one)
                if i < 4:
                    print(f'🎨 Waiting 5 seconds before next upscale request...')
                    await asyncio.sleep(5)
            
            # Step 3: Wait for all upscale tasks to complete and collect results
            results = []
            for i, upscale_task_id in enumerate(upscale_task_ids, 1):
                print(f'🎨 Step 3.{i}: Waiting for upscale task {i}/4 to complete: {upscale_task_id}')
                upscale_task_data = await self._wait_for_task_completion(upscale_task_id)
                
                # Process and save upscaled image
                upscale_result = await self._process_task_result(upscale_task_data)
                results.append(upscale_result)
                
                print(f'🎨 Upscaled image {i}/4 completed: {upscale_result[3]}')
            
            print(f'🎨 All 4 upscaled images completed!')
            return results
            
        except Exception as e:
            print(f"🎨 Midjourney upscale generation error: {str(e)}")
            print(traceback.format_exc())
            raise e

    async def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str = "1:1",
        input_images: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[str, int, int, str]:
        """
        Generate image using Midjourney Proxy API
        
        Args:
            prompt: Image generation prompt
            model: Model name (not used for Midjourney)
            aspect_ratio: Image aspect ratio
            input_images: Optional input images for reference
            metadata: Optional metadata to be saved
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, int, int, str]: (mime_type, width, height, filename)
        """
        try:
            # Convert aspect ratio to Midjourney format
            aspect_ratio_map = {
                "1:1": " --ar 1:1",
                "16:9": " --ar 16:9", 
                "9:16": " --ar 9:16",
                "4:3": " --ar 4:3",
                "3:4": " --ar 3:4"
            }
            
            # Add aspect ratio to prompt if specified
            enhanced_prompt = prompt
            if aspect_ratio in aspect_ratio_map:
                enhanced_prompt += aspect_ratio_map[aspect_ratio]
            
            # Convert input images to base64 if provided
            base64_images = None
            if input_images:
                base64_images = []
                for image_path in input_images:
                    if os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            import base64
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                            # Determine image format
                            if image_path.lower().endswith('.png'):
                                base64_images.append(f"data:image/png;base64,{image_data}")
                            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                                base64_images.append(f"data:image/jpeg;base64,{image_data}")
                            else:
                                base64_images.append(f"data:image/png;base64,{image_data}")
            
            # Submit task
            task_id = await self._submit_imagine_task(enhanced_prompt, base64_images)
            
            # Wait for completion
            task_data = await self._wait_for_task_completion(task_id)
            
            # Process result
            return await self._process_task_result(task_data)
            
        except Exception as e:
            print(f"🎨 Midjourney generation error: {str(e)}")
            print(traceback.format_exc())
            raise e