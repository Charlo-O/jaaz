# services/OpenAIAgents_service/jaaz_agent.py

from typing import Dict, Any, List
import asyncio
import os
from nanoid import generate
from tools.utils.image_canvas_utils import save_image_to_canvas
from tools.utils.image_utils import get_image_info_and_save
from services.config_service import FILES_DIR
from common import DEFAULT_PORT
from ..jaaz_service import JaazService
from ..multimodal_ai_service import multimodal_ai_service


async def create_jaaz_response(messages: List[Dict[str, Any]], session_id: str = "", canvas_id: str = "") -> Dict[str, Any]:
    """
    基于云端服务的图像生成响应函数
    实现和 magic_agent 相同的功能
    """
    try:
        # 获取图片和视频内容
        user_message: Dict[str, Any] = messages[-1]
        image_content: str = ""
        video_content: str = ""
        media_type: str = ""

        if isinstance(user_message.get('content'), list):
            for content_item in user_message['content']:
                if content_item.get('type') == 'image_url':
                    image_content = content_item.get(
                        'image_url', {}).get('url', "")
                    media_type = "image"
                    break
                elif content_item.get('type') == 'video_url':
                    video_content = content_item.get(
                        'video_url', {}).get('url', "")
                    media_type = "video"
                    break

        if not image_content and not video_content:
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '✨ not found input image or video'
                    }
                ]
            }

        # 创建 Jaaz 服务实例
        try:
            jaaz_service = JaazService()
        except ValueError as e:
            print(f"❌ Jaaz service configuration error: {e}")
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '✨ Cloud API Key not configured'
                    }
                ]
            }

        # 调用 Jaaz 服务生成魔法图像
        result = await jaaz_service.generate_magic_image(image_content)
        if not result:
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '✨ Magic generation failed'
                    }
                ]
            }

        # 检查是否有错误
        if result.get('error'):
            error_msg = result['error']
            print(f"❌ Magic generation error: {error_msg}")
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': f'✨ Magic Generation Error: {error_msg}'
                    }
                ]
            }

        # 检查是否有结果 URL
        if not result.get('result_url'):
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '✨ Magic generation failed: No result URL'
                    }
                ]
            }

        # 初始化变量
        filename = ""
        image_url = result['result_url']

        # 保存图片到画布
        if session_id and canvas_id:
            try:
                # 生成唯一文件名
                file_id = generate(size=10)
                file_path_without_extension = os.path.join(FILES_DIR, file_id)

                # 下载并保存图片
                mime_type, width, height, extension = await get_image_info_and_save(
                    image_url, file_path_without_extension, is_b64=False
                )

                width = max(1, int(width / 2))
                height = max(1, int(height / 2))

                # 生成文件名
                filename = f'{file_id}.{extension}'

                # 保存图片到画布
                image_url = await save_image_to_canvas(session_id, canvas_id, filename, mime_type, width, height)
                print(f"✨ 图片已保存到画布: {filename}")
            except Exception as e:
                print(f"❌ 保存图片到画布失败: {e}")

        return {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': f'✨ Magic Success!!! ![image_id: {filename}](http://localhost:{DEFAULT_PORT}{image_url})'
                },
            ]
        }

    except (asyncio.TimeoutError, Exception) as e:
        # 检查是否是超时相关的错误
        error_msg = str(e).lower()
        if 'timeout' in error_msg or 'timed out' in error_msg:
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '✨ time out'
                    }
                ]
            }
        else:
            print(f"❌ 创建魔法回复时出错: {e}")
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': f'✨ Magic Generation Error: {str(e)}'
                    }
                ]
            }


async def create_multimodal_analysis_response(
    messages: List[Dict[str, Any]], 
    session_id: str = "", 
    canvas_id: str = ""
) -> Dict[str, Any]:
    """
    处理图片和视频的AI分析响应函数
    支持多种AI模型（OpenAI, Google Gemini等）
    """
    try:
        # 获取图片和视频内容
        user_message: Dict[str, Any] = messages[-1]
        media_content: str = ""
        media_type: str = ""
        text_prompt: str = ""

        # 提取文本提示
        if isinstance(user_message.get('content'), list):
            for content_item in user_message['content']:
                if content_item.get('type') == 'text':
                    text_prompt = content_item.get('text', '')
                elif content_item.get('type') == 'image_url':
                    media_content = content_item.get('image_url', {}).get('url', "")
                    media_type = "image"
                elif content_item.get('type') == 'video_url':
                    media_content = content_item.get('video_url', {}).get('url', "")
                    media_type = "video"

        if not media_content:
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': '✨ 未找到图片或视频内容进行分析'
                    }
                ]
            }

        # 准备分析提示词
        if not text_prompt:
            if media_type == "image":
                text_prompt = "请详细描述这张图片的内容，包括场景、物体、人物、颜色、文字等信息。"
            else:  # video
                text_prompt = "请详细描述这个视频的内容，包括场景、动作、人物、物体、情节发展等信息。"
        
        print(f"🎬 开始分析{media_type}内容...")

        # 调用多模态AI服务进行分析
        analysis_result = await multimodal_ai_service.analyze_media(
            media_content=media_content,
            media_type=media_type,
            prompt=text_prompt,
            preferred_provider="openai",  # 优先使用OpenAI
            preferred_model=""  # 自动选择合适的模型
        )

        if analysis_result.get('success'):
            result_text = analysis_result['result']
            provider = analysis_result.get('provider', 'unknown')
            model = analysis_result.get('model', 'unknown')
            
            response_text = f"✨ {media_type.upper()}分析结果 (由 {provider}/{model} 提供):\n\n{result_text}"
            
            print(f"✅ {media_type}分析完成: {provider}/{model}")
            
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': response_text
                    }
                ]
            }
        else:
            error_msg = analysis_result.get('error', 'Unknown error')
            print(f"❌ {media_type}分析失败: {error_msg}")
            
            return {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': f'✨ {media_type.upper()}分析失败: {error_msg}'
                    }
                ]
            }

    except Exception as e:
        print(f"❌ 创建多模态分析回复时出错: {e}")
        return {
            'role': 'assistant',
            'content': [
                {
                    'type': 'text',
                    'text': f'✨ 多模态分析错误: {str(e)}'
                }
            ]
        }


if __name__ == "__main__":
    asyncio.run(create_jaaz_response([]))
