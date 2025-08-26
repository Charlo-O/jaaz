# services/websocket_service.py
from services.websocket_state import sio, get_all_socket_ids
import traceback
from typing import Any, Dict


def _filter_base64_from_data(data: Any) -> Any:
    """过滤数据中的base64编码，避免在WebSocket传输中显示长串编码

    注意：此函数仅用于过滤WebSocket输出显示，不影响实际业务逻辑
    过滤较小的base64数据以避免终端显示混乱
    """
    if isinstance(data, dict):
        filtered = {}
        for key, value in data.items():
            # 与全局过滤器保持一致，过滤超过100KB的数据
            if isinstance(value, str) and len(value) > 100 * 1024:  # 100KB
                filtered[key] = f"[filtered large data - {len(value)} chars]"
            elif isinstance(value, (dict, list)):
                filtered[key] = _filter_base64_from_data(value)
            else:
                filtered[key] = value
        return filtered
    elif isinstance(data, list):
        return [_filter_base64_from_data(item) for item in data]
    else:
        return data


async def broadcast_session_update(
    session_id: str, canvas_id: str | None, event: Dict[str, Any]
):
    socket_ids = get_all_socket_ids()
    if socket_ids:
        try:
            # 过滤事件数据中的base64编码
            filtered_event = _filter_base64_from_data(event)
            message_data = {
                'canvas_id': canvas_id,
                'session_id': session_id,
                **filtered_event,
            }

            for socket_id in socket_ids:
                await sio.emit('session_update', message_data, room=socket_id)
        except Exception as e:
            print(f"Error broadcasting session update for {session_id}: {e}")
            traceback.print_exc()


# compatible with legacy codes
# TODO: All Broadcast should have a canvas_id


async def send_to_websocket(session_id: str, event: Dict[str, Any]):
    # 过滤事件数据中的base64编码
    filtered_event = _filter_base64_from_data(event)
    await broadcast_session_update(session_id, None, filtered_event)


async def broadcast_init_done():
    try:
        await sio.emit('init_done', {'type': 'init_done'})
        print("Broadcasted init_done to all clients")
    except Exception as e:
        print(f"Error broadcasting init_done: {e}")
        traceback.print_exc()
