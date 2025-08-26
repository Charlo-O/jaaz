# type: ignore[import]
import traceback
<<<<<<< Updated upstream
from typing import Optional, List, Dict, Any, Callable, Awaitable
from langchain_core.messages import AIMessageChunk, ToolCall, convert_to_openai_messages, ToolMessage
=======
from typing import Optional, List, Dict, Any, Callable, Awaitable, Set
from langchain_core.messages import (
    AIMessageChunk,
    ToolCall,
    convert_to_openai_messages,
    ToolMessage,
)
>>>>>>> Stashed changes
from langgraph.graph import StateGraph
import json


class StreamProcessor:
    """流式处理器 - 负责处理智能体的流式输出"""

    def __init__(
        self,
        session_id: str,
        db_service: Any,
        websocket_service: Callable[[str, Dict[str, Any]], Awaitable[None]],
    ):
        self.session_id = session_id
        self.db_service = db_service
        self.websocket_service = websocket_service
        self.tool_calls: List[ToolCall] = []
        self.last_saved_message_index = 0
        self.last_streaming_tool_call_id: Optional[str] = None

    def _filter_base64_from_message(self, message):
        """过滤消息中的 base64 数据，避免在日志中显示

        注意：此函数仅用于过滤终端输出显示，不影响实际业务逻辑
        只过滤非常大的数据以避免终端显示混乱，但保留业务必需的数据
        """
        if isinstance(message, dict):
            filtered = {}
            for key, value in message.items():
                # 对于输出过滤，使用100KB阈值（与全局过滤器保持一致）
                if isinstance(value, str) and len(value) > 100 * 1024:  # 100KB
                    filtered[key] = f"[filtered base64 data - {len(value)} chars]"
                elif isinstance(value, (dict, list)):
                    filtered[key] = self._filter_base64_from_message(value)
                else:
                    filtered[key] = value
            return filtered
        elif isinstance(message, list):
            return [self._filter_base64_from_message(item) for item in message]
        else:
            return message

    async def process_stream(
        self, swarm: StateGraph, messages: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> None:
        """处理整个流式响应

        Args:
            swarm: 智能体群组
            messages: 消息列表
            context: 上下文信息
        """
        self.last_saved_message_index = len(messages) - 1

        compiled_swarm = swarm.compile()

        async for chunk in compiled_swarm.astream(
            {"messages": messages},
            config=context,
            stream_mode=["messages", "custom", 'values'],
        ):
            await self._handle_chunk(chunk)

        # 发送完成事件
        await self.websocket_service(self.session_id, {'type': 'done'})

    async def _handle_chunk(self, chunk: Any) -> None:
        # print('👇chunk', chunk)
        """处理单个chunk"""
        chunk_type = chunk[0]

        if chunk_type == 'values':
            await self._handle_values_chunk(chunk[1])
        else:
            await self._handle_message_chunk(chunk[1][0])

    async def _handle_values_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """处理 values 类型的 chunk"""
        all_messages = chunk_data.get('messages', [])
        oai_messages = convert_to_openai_messages(all_messages)
        # 确保 oai_messages 是列表类型
        if not isinstance(oai_messages, list):
            oai_messages = [oai_messages] if oai_messages else []

        # 发送所有消息到前端（过滤base64数据）
        filtered_messages = [
            self._filter_base64_from_message(msg) for msg in oai_messages
        ]
        await self.websocket_service(
            self.session_id, {'type': 'all_messages', 'messages': filtered_messages}
        )

        # 保存新消息到数据库
        for i in range(self.last_saved_message_index + 1, len(oai_messages)):
            new_message = oai_messages[i]
            if len(oai_messages) > 0:  # 确保有消息才保存
                await self.db_service.create_message(
                    self.session_id,
                    new_message.get('role', 'user'),
                    json.dumps(new_message),
                )
            self.last_saved_message_index = i

    async def _handle_message_chunk(self, ai_message_chunk: AIMessageChunk) -> None:
        """处理消息类型的 chunk"""
        # print('👇ai_message_chunk', ai_message_chunk)
        try:
            content = ai_message_chunk.content

            if isinstance(ai_message_chunk, ToolMessage):
                # 工具调用结果之后会在 values 类型中发送到前端，这里会更快出现一些
                oai_message = convert_to_openai_messages([ai_message_chunk])[0]
                # 过滤 base64 数据后打印消息
                filtered_message = self._filter_base64_from_message(oai_message)
                print('👇toolcall res oai_message', filtered_message)
                # 过滤消息中的base64数据再发送
                filtered_oai_message = self._filter_base64_from_message(oai_message)
                await self.websocket_service(
                    self.session_id,
                    {
                        'type': 'tool_call_result',
                        'id': ai_message_chunk.tool_call_id,
                        'message': filtered_oai_message,
                    },
                )
            elif content:
                # 发送文本内容
                await self.websocket_service(
                    self.session_id, {'type': 'delta', 'text': content}
                )
            elif (
                hasattr(ai_message_chunk, 'tool_calls')
                and ai_message_chunk.tool_calls
                and ai_message_chunk.tool_calls[0].get('name')
            ):
                # 处理工具调用
                await self._handle_tool_calls(ai_message_chunk.tool_calls)

            # 处理工具调用参数流
            if hasattr(ai_message_chunk, 'tool_call_chunks'):
                await self._handle_tool_call_chunks(ai_message_chunk.tool_call_chunks)
        except Exception as e:
            print('🟠error', e)
            traceback.print_stack()

    async def _handle_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """处理工具调用"""
        self.tool_calls = [tc for tc in tool_calls if tc.get('name')]
        print('😘tool_call event', tool_calls)

        # 需要确认的工具列表
        TOOLS_REQUIRING_CONFIRMATION = {
            # 'generate_video_by_kling_v2_jaaz',
            # 'generate_video_by_seedance_v1_pro_volces',
            # 'generate_video_by_seedance_v1_lite_i2v',
            # 'generate_video_by_seedance_v1_lite_t2v',
            # 'generate_video_by_seedance_v1_jaaz',
            # 'generate_video_by_hailuo_02_jaaz',
            'generate_video_by_veo3_fast_jaaz',
        }

        for tool_call in self.tool_calls:
            tool_name = tool_call.get('name')

            # 检查是否需要确认
            if tool_name in TOOLS_REQUIRING_CONFIRMATION:
                # 对于需要确认的工具，不在这里发送事件，让工具函数自己处理
                print(
                    f'🔄 Tool {tool_name} requires confirmation, skipping StreamProcessor event'
                )
                continue
            else:
<<<<<<< Updated upstream
                await self.websocket_service(self.session_id, {
                    'type': 'tool_call',
                    'id': tool_call.get('id'),
                    'name': tool_name,
                    'arguments': '{}'
                })
=======
                await self.websocket_service(
                    self.session_id,
                    {
                        'type': 'tool_call',
                        'id': tool_id,
                        'name': tool_name,
                        'arguments': '{}',
                    },
                )
>>>>>>> Stashed changes

    async def _handle_tool_call_chunks(self, tool_call_chunks: List[Any]) -> None:
        """处理工具调用参数流"""
        for tool_call_chunk in tool_call_chunks:
            if tool_call_chunk.get('id'):
                # 标记新的流式工具调用参数开始
                self.last_streaming_tool_call_id = tool_call_chunk.get('id')
            else:
                if self.last_streaming_tool_call_id:
                    await self.websocket_service(
                        self.session_id,
                        {
                            'type': 'tool_call_arguments',
                            'id': self.last_streaming_tool_call_id,
                            'text': tool_call_chunk.get('args'),
                        },
                    )
                else:
                    print('🟠no last_streaming_tool_call_id', tool_call_chunk)
