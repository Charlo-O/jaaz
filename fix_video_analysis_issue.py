#!/usr/bin/env python3
"""
视频分析问题诊断和修复脚本
用于解决画布显示base64编码图表而不是关键帧的问题
"""

import os
import json
import sqlite3
from pathlib import Path

# 配置
WORKSPACE_ROOT = Path(__file__).parent
FILES_DIR = WORKSPACE_ROOT / "server" / "files"
DATABASE_PATH = WORKSPACE_ROOT / "server" / "database" / "database.db"


def diagnose_canvas_issue():
    """诊断画布中的图片问题"""
    print("🔍 开始诊断画布图片显示问题...")

    # 1. 检查FILES_DIR中的文件
    print(f"\n📁 检查文件目录: {FILES_DIR}")
    if not FILES_DIR.exists():
        print(f"❌ 文件目录不存在: {FILES_DIR}")
        return

    # 查找关键帧文件
    keyframe_files = list(FILES_DIR.glob("*_keyframe_*.png"))
    matplotlib_files = list(FILES_DIR.glob("*chart*.png")) + list(
        FILES_DIR.glob("*graph*.png")
    )

    print(f"✅ 找到 {len(keyframe_files)} 个关键帧文件")
    for f in keyframe_files[:5]:  # 只显示前5个
        print(f"   - {f.name}")

    print(f"⚠️ 找到 {len(matplotlib_files)} 个可能的图表文件")
    for f in matplotlib_files[:5]:  # 只显示前5个
        print(f"   - {f.name}")

    # 2. 检查数据库中的画布数据
    print(f"\n🗄️ 检查数据库: {DATABASE_PATH}")
    if not DATABASE_PATH.exists():
        print(f"❌ 数据库不存在: {DATABASE_PATH}")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # 查询画布数据
        cursor.execute(
            "SELECT id, name, data FROM canvases ORDER BY updated_at DESC LIMIT 5"
        )
        canvases = cursor.fetchall()

        print(f"📊 找到 {len(canvases)} 个最近的画布")

        for canvas_id, canvas_name, canvas_data in canvases:
            print(f"\n🎨 画布: {canvas_name} (ID: {canvas_id})")

            if canvas_data:
                try:
                    data = json.loads(canvas_data)
                    elements = data.get('elements', [])
                    files = data.get('files', {})

                    image_elements = [e for e in elements if e.get('type') == 'image']
                    print(f"   - 图片元素数量: {len(image_elements)}")
                    print(f"   - 文件数量: {len(files)}")

                    # 检查文件类型
                    for file_id, file_info in list(files.items())[:3]:  # 只检查前3个
                        file_url = file_info.get('dataURL', '')
                        mime_type = file_info.get('mimeType', '')
                        print(f"   - 文件: {file_url} ({mime_type})")

                        # 检查是否为关键帧文件
                        if 'keyframe' in file_url:
                            print(f"     ✅ 这是关键帧文件")
                        elif 'chart' in file_url or 'graph' in file_url:
                            print(f"     ⚠️ 这可能是图表文件")
                        elif file_url.startswith('data:'):
                            print(f"     ❌ 这是base64编码的文件（可能是问题所在）")

                except json.JSONDecodeError:
                    print(f"   ❌ 画布数据解析失败")

        conn.close()

    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")


def suggest_fixes():
    """提供修复建议"""
    print("\n💡 修复建议:")

    suggestions = [
        "1. 确认前端调用的是正确的API:",
        "   - 应该调用: /api/analyze_video_to_canvas",
        "   - 而不是: /api/process_video",
        "",
        "2. 使用诊断工具进行测试:",
        "   - 访问: http://localhost:57988/debug_video_analysis.html",
        "   - 按步骤进行完整的诊断",
        "",
        "3. 检查控制台日志:",
        "   - 查看前端开发者工具中的网络请求",
        "   - 确认调用的API端点",
        "",
        "4. 清除旧的画布数据:",
        "   - 删除包含base64编码图片的画布元素",
        "   - 重新上传视频并分析",
        "",
        "5. 验证文件生成:",
        f"   - 检查 {FILES_DIR} 目录",
        "   - 确认生成了 *_keyframe_*.png 文件",
    ]

    for suggestion in suggestions:
        print(suggestion)


def clean_base64_images():
    """清理画布中的base64编码图片"""
    print("\n🧹 清理画布中的base64编码图片...")

    if not DATABASE_PATH.exists():
        print(f"❌ 数据库不存在: {DATABASE_PATH}")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # 获取所有画布
        cursor.execute("SELECT id, name, data FROM canvases")
        canvases = cursor.fetchall()

        cleaned_count = 0

        for canvas_id, canvas_name, canvas_data in canvases:
            if not canvas_data:
                continue

            try:
                data = json.loads(canvas_data)
                files = data.get('files', {})
                elements = data.get('elements', [])

                # 找到base64编码的文件
                base64_file_ids = []
                for file_id, file_info in files.items():
                    data_url = file_info.get('dataURL', '')
                    if data_url.startswith('data:'):
                        base64_file_ids.append(file_id)

                if base64_file_ids:
                    print(
                        f"🎨 画布 {canvas_name}: 找到 {len(base64_file_ids)} 个base64文件"
                    )

                    # 删除base64文件
                    for file_id in base64_file_ids:
                        del files[file_id]

                    # 删除对应的元素
                    data['elements'] = [
                        e for e in elements if e.get('fileId') not in base64_file_ids
                    ]

                    # 更新数据库
                    cursor.execute(
                        "UPDATE canvases SET data = ?, updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                        (json.dumps(data), canvas_id),
                    )

                    cleaned_count += 1
                    print(f"   ✅ 已清理 {len(base64_file_ids)} 个base64文件")

            except json.JSONDecodeError:
                print(f"   ❌ 画布数据解析失败: {canvas_name}")

        conn.commit()
        conn.close()

        print(f"\n✅ 清理完成，共处理 {cleaned_count} 个画布")

    except Exception as e:
        print(f"❌ 清理失败: {e}")


def main():
    """主函数"""
    print("🔧 视频分析问题诊断和修复工具")
    print("=" * 50)

    # 诊断问题
    diagnose_canvas_issue()

    # 提供建议
    suggest_fixes()

    # 询问是否清理
    print("\n" + "=" * 50)
    response = input("是否要清理画布中的base64编码图片? (y/N): ").strip().lower()

    if response in ['y', 'yes']:
        clean_base64_images()
        print("\n🎉 建议现在重新上传视频并使用正确的API进行分析")
    else:
        print(
            "\n💡 请手动使用诊断工具: http://localhost:57988/debug_video_analysis.html"
        )


if __name__ == "__main__":
    main()
