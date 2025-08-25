import asyncio
import time
from typing import List, Dict, Any, Optional, Sequence
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool
import httpx
from utils.http_client import HttpClient
from services.settings_service import settings_service


class CustomGeminiChat(BaseChatModel):
    """自定义 Gemini 聊天模型，使用统一 httpx 客户端并支持代理配置"""

    api_key: str
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    bound_tools: List[Dict[str, Any]] = []

    # 使用 Pydantic 自动生成的 __init__，无需重写

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

    def _convert_messages_to_gemini_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """转换 LangChain 消息格式为 Gemini 格式"""
        gemini_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, (HumanMessage, SystemMessage)):
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": msg.content}]}
                )
            elif isinstance(msg, AIMessage):
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": msg.content}]}
                )

        return gemini_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成响应"""
        # 转换消息格式
        gemini_messages = self._convert_messages_to_gemini_format(messages)

        # 确保模型名称不包含多余的 "models/" 前缀
        model_name = self.model_name
        if model_name.startswith("models/"):
            model_name = model_name[7:]  # 移除 "models/" 前缀

        # 构建请求
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

        data = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 8000,
            },
        }

        # 如果有绑定的工具，添加到请求中
        if self.bound_tools:
            data["tools"] = [{"functionDeclarations": self.bound_tools}]

        # 准备代理设置：支持 settings.json 的 proxy 配置
        proxy_setting: str = str(settings_service.get_proxy_config() or 'system')
        proxies_arg: Any = None
        if proxy_setting == 'no_proxy':
            # 显式禁用代理
            proxies_arg = {}
        elif proxy_setting in ('', 'system'):
            # 使用系统/环境代理，不传 proxies
            proxies_arg = None
        elif proxy_setting.startswith(
            ('http://', 'https://', 'socks4://', 'socks5://')
        ):
            # 使用自定义代理 URL（httpx 原生支持 socks，需要 socksio 依赖，已在 requirements 中）
            proxies_arg = proxy_setting

        try:
            # 使用重试机制
            for attempt in range(3):
                try:
                    timeout_seconds = 60 if self.bound_tools else 30
                    timeout = httpx.Timeout(timeout_seconds)

                    # 按代理模式发起请求（避免 httpx 版本差异带来的 proxies 参数不兼容）
                    if proxies_arg is None:
                        # 使用系统/环境代理
                        with HttpClient.create_sync(timeout=timeout) as client:
                            response = client.post(
                                url,
                                json=data,
                                headers={"Content-Type": "application/json"},
                            )
                    elif proxies_arg == {}:
                        # 显式禁用代理：关闭对环境变量的信任
                        with HttpClient.create_sync(
                            timeout=timeout, trust_env=False
                        ) as client:
                            response = client.post(
                                url,
                                json=data,
                                headers={"Content-Type": "application/json"},
                            )
                    else:
                        # 自定义代理：临时注入环境变量（兼容不同 httpx 版本）
                        prev_http = os.environ.get('HTTP_PROXY')
                        prev_https = os.environ.get('HTTPS_PROXY')
                        prev_all = os.environ.get('ALL_PROXY')
                        prev_http_l = os.environ.get('http_proxy')
                        prev_https_l = os.environ.get('https_proxy')
                        prev_all_l = os.environ.get('all_proxy')
                        try:
                            os.environ['HTTP_PROXY'] = str(proxies_arg)
                            os.environ['HTTPS_PROXY'] = str(proxies_arg)
                            os.environ['ALL_PROXY'] = str(proxies_arg)
                            os.environ['http_proxy'] = str(proxies_arg)
                            os.environ['https_proxy'] = str(proxies_arg)
                            os.environ['all_proxy'] = str(proxies_arg)
                            with HttpClient.create_sync(
                                timeout=timeout, trust_env=True
                            ) as client:
                                response = client.post(
                                    url,
                                    json=data,
                                    headers={"Content-Type": "application/json"},
                                )
                        finally:
                            if prev_http is None:
                                os.environ.pop('HTTP_PROXY', None)
                            else:
                                os.environ['HTTP_PROXY'] = prev_http
                            if prev_https is None:
                                os.environ.pop('HTTPS_PROXY', None)
                            else:
                                os.environ['HTTPS_PROXY'] = prev_https
                            if prev_all is None:
                                os.environ.pop('ALL_PROXY', None)
                            else:
                                os.environ['ALL_PROXY'] = prev_all
                            if prev_http_l is None:
                                os.environ.pop('http_proxy', None)
                            else:
                                os.environ['http_proxy'] = prev_http_l
                            if prev_https_l is None:
                                os.environ.pop('https_proxy', None)
                            else:
                                os.environ['https_proxy'] = prev_https_l
                            if prev_all_l is None:
                                os.environ.pop('all_proxy', None)
                            else:
                                os.environ['all_proxy'] = prev_all_l

                    if response.status_code == 200:
                        result = response.json()
                        if "candidates" in result and len(result["candidates"]) > 0:
                            candidate = result["candidates"][0]
                            content = self._extract_content_from_candidate(candidate)
                            message = AIMessage(content=content)
                            generation = ChatGeneration(message=message)
                            return ChatResult(generations=[generation])

                    error_msg = (
                        f"API call failed: {response.status_code} - {response.text}"
                    )
                    if attempt == 2:
                        raise Exception(error_msg)
                    print(f"⚠️ 尝试 {attempt + 1} 失败，正在重试...")
                    time.sleep(1)
                except (
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    httpx.ConnectError,
                    httpx.ProxyError,
                ) as e:
                    if attempt == 2:
                        proxy_desc = proxy_setting if proxy_setting else 'system'
                        raise Exception(
                            f"Network timeout after 3 attempts: {str(e)} | proxy={proxy_desc}"
                        )
                    print(f"⚠️ 网络超时，尝试 {attempt + 1}/3，正在重试...")
                    time.sleep(2)
                except Exception:
                    # 非网络类错误直接抛出
                    raise
        except Exception as e:
            raise Exception(f"Custom Gemini call failed: {str(e)}")

        # 保底抛错，保证类型检查通过
        raise Exception("Custom Gemini call failed: unreachable state")

    def _extract_content_from_candidate(self, candidate: Dict[str, Any]) -> str:
        """从候选响应中提取内容，处理不同的响应格式"""
        try:
            # 尝试标准格式
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0:
                    # 处理文本响应
                    if "text" in parts[0]:
                        return parts[0]["text"]
                    # 处理函数调用响应
                    elif "functionCall" in parts[0]:
                        func_call = parts[0]["functionCall"]
                        return f"调用函数: {func_call.get('name', 'unknown')} 参数: {func_call.get('args', {})}"

            # 如果以上都失败，尝试其他可能的格式
            if isinstance(candidate, dict):
                # 查找任何包含 text 的字段
                for _, value in candidate.items():
                    if isinstance(value, dict) and "text" in value:
                        return value["text"]
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "text" in item:
                                return item["text"]

            # 如果仍然找不到内容，返回完整的候选对象用于调试
            return f"未知响应格式: {candidate}"

        except Exception as e:
            return f"解析响应时出错: {str(e)}, 原始数据: {candidate}"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步版本 - 在线程池中运行同步代码"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._generate(messages, stop, None, **kwargs)
        )

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> "CustomGeminiChat":
        """绑定工具到模型"""
        # 创建一个新的实例，保留当前的配置
        bound_model = CustomGeminiChat(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            bound_tools=[],
            **kwargs,
        )

        # 转换工具格式（简化处理，避免复杂 schema 导致的兼容问题）
        formatted_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                try:
                    name = getattr(tool, 'name', 'tool')
                    description = getattr(tool, 'description', '')
                    formatted_tools.append(
                        {
                            "name": name,
                            "description": description,
                            "parameters": {"type": "object"},
                        }
                    )
                except Exception:
                    continue

        bound_model.bound_tools = formatted_tools
        return bound_model

    def _convert_tool_to_gemini_format(self, tool: BaseTool) -> Dict[str, Any]:
        """将 LangChain 工具转换为 Gemini 兼容格式（简化版）"""
        name = getattr(tool, 'name', 'tool')
        description = getattr(tool, 'description', '')
        return {
            "name": name,
            "description": description,
            "parameters": {"type": "object"},
        }

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "CustomGeminiChat":
        """支持结构化输出（简单实现）"""
        return self
