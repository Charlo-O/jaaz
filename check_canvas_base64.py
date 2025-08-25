#!/usr/bin/env python3
"""
检查画布数据中的base64编码问题
"""

import sqlite3
import json
import re
import os


def main():
    db_path = "server/app.db"

    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 查询所有画布数据
        cursor.execute("SELECT id, name, data FROM canvases")
        canvases = cursor.fetchall()

        print(f"📊 找到 {len(canvases)} 个画布")
        print("=" * 60)

        for canvas_id, canvas_name, canvas_data_str in canvases:
            print(f"\n🎨 画布: {canvas_name} (ID: {canvas_id})")

            if not canvas_data_str:
                print("   ⚠️ 画布数据为空")
                continue

            try:
                data = json.loads(canvas_data_str)
                files = data.get('files', {})
                elements = data.get('elements', [])

                print(f"   📁 文件数量: {len(files)}")
                print(f"   🔲 元素数量: {len(elements)}")

                # 检查文件中的base64内容
                base64_files = []
                http_files = []

                for file_id, file_info in files.items():
                    data_url = file_info.get('dataURL', '')
                    mime_type = file_info.get('mimeType', '')

                    if data_url.startswith('data:'):
                        base64_files.append(
                            {
                                'id': file_id,
                                'url': (
                                    data_url[:100] + '...'
                                    if len(data_url) > 100
                                    else data_url
                                ),
                                'mime_type': mime_type,
                                'size': len(data_url),
                            }
                        )
                    elif data_url.startswith('/api/file/') or data_url.startswith(
                        'http'
                    ):
                        http_files.append(
                            {'id': file_id, 'url': data_url, 'mime_type': mime_type}
                        )

                if base64_files:
                    print(f"   🔴 发现 {len(base64_files)} 个base64文件:")
                    for f in base64_files:
                        print(
                            f"      - {f['id']}: {f['mime_type']} ({f['size']} chars)"
                        )
                        print(f"        数据: {f['url']}")

                if http_files:
                    print(f"   ✅ 正常HTTP文件 {len(http_files)} 个:")
                    for f in http_files:
                        print(f"      - {f['id']}: {f['mime_type']} -> {f['url']}")

                # 检查元素类型
                element_types = {}
                for element in elements:
                    elem_type = element.get('type', 'unknown')
                    element_types[elem_type] = element_types.get(elem_type, 0) + 1

                if element_types:
                    print(f"   📋 元素类型统计: {element_types}")

            except json.JSONDecodeError:
                print(f"   ❌ 画布数据解析失败")

        conn.close()

        print("\n" + "=" * 60)
        print("🔍 建议的修复操作:")
        print("1. 如果发现base64文件，运行: python fix_video_analysis_issue.py")
        print("2. 重启服务器应用最新修复")
        print("3. 刷新前端页面清除缓存")

    except Exception as e:
        print(f"❌ 检查失败: {e}")


if __name__ == "__main__":
    main()
