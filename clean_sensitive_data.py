#!/usr/bin/env python3
"""
敏感数据清理脚本
用于清理项目中可能包含的API密钥、密码等敏感信息
"""
import os
import shutil
import json
import toml
from pathlib import Path


def clean_config_files():
    """清理配置文件中的API密钥"""
    print("🧹 清理配置文件...")

    # 清理 config.toml
    config_path = "server/user_data/config.toml"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = toml.load(f)

            # 清空所有API密钥
            for provider in config.values():
                if isinstance(provider, dict) and "api_key" in provider:
                    provider["api_key"] = ""

            with open(config_path, "w", encoding="utf-8") as f:
                toml.dump(config, f)

            print(f"✅ 已清理 {config_path}")
        except Exception as e:
            print(f"❌ 清理 {config_path} 失败: {e}")


def clean_database():
    """删除数据库文件（包含聊天历史）"""
    print("🧹 清理数据库...")

    db_path = "server/user_data/localmanus.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"✅ 已删除 {db_path}")
        except Exception as e:
            print(f"❌ 删除 {db_path} 失败: {e}")
    else:
        print(f"ℹ️ 数据库文件不存在: {db_path}")


def clean_user_files():
    """清理用户上传的文件"""
    print("🧹 清理用户文件...")

    files_dir = "server/user_data/files"
    if os.path.exists(files_dir):
        try:
            # 获取文件数量
            file_count = len(
                [
                    f
                    for f in os.listdir(files_dir)
                    if os.path.isfile(os.path.join(files_dir, f))
                ]
            )

            # 清空目录
            shutil.rmtree(files_dir)
            os.makedirs(files_dir)

            print(f"✅ 已清理 {file_count} 个用户文件")
        except Exception as e:
            print(f"❌ 清理用户文件失败: {e}")
    else:
        print(f"ℹ️ 用户文件目录不存在: {files_dir}")


def clean_logs():
    """清理日志文件"""
    print("🧹 清理日志文件...")

    log_files = ["network_diagnosis.log", "server.log", "debug.log"]

    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                print(f"✅ 已删除 {log_file}")
            except Exception as e:
                print(f"❌ 删除 {log_file} 失败: {e}")


def clean_temp_files():
    """清理临时文件"""
    print("🧹 清理临时文件...")

    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*.cache",
        "__pycache__",
        ".pytest_cache",
        "node_modules/.cache",
    ]

    for pattern in temp_patterns:
        # 这里可以添加具体的清理逻辑
        pass


def main():
    """主函数"""
    print("🔒 开始清理敏感数据...")
    print("=" * 50)

    # 确保在项目根目录执行
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    try:
        clean_config_files()
        clean_database()
        clean_user_files()
        clean_logs()
        clean_temp_files()

        print("=" * 50)
        print("✅ 敏感数据清理完成！")
        print()
        print("⚠️ 注意事项：")
        print("1. 所有API密钥已被清空，需要重新配置")
        print("2. 聊天历史和用户文件已被删除")
        print("3. 请检查代码中是否还有硬编码的敏感信息")

    except Exception as e:
        print(f"❌ 清理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
