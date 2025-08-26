"""
全局终端输出过滤器
用于防止base64编码在终端中显示
"""

import sys
import io
import re
from typing import Any


class Base64Filter:
    """过滤base64编码的输出过滤器"""

    def __init__(self, original_stream):
        self.original_stream = original_stream
        # 过滤base64数据URL (>100KB)
        self.base64_pattern = re.compile(
            r'data:[^;]+;base64,[A-Za-z0-9+/=]{102400,}'
        )  # 100KB
        # 过滤长字符串 (>500KB)
        self.long_string_pattern = re.compile(r'[A-Za-z0-9+/=]{512000,}')  # 500KB

    def write(self, text: str) -> int:
        """过滤并写入文本"""
        if isinstance(text, str):
            # 过滤base64数据URL
            text = self.base64_pattern.sub(
                lambda m: f"[filtered base64 data - {len(m.group())} chars]", text
            )

            # 过滤长字符串
            text = self.long_string_pattern.sub(
                lambda m: f"[filtered long string - {len(m.group())} chars]", text
            )

        return self.original_stream.write(text)

    def flush(self):
        """刷新输出"""
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()

    def __getattr__(self, name):
        """代理其他属性到原始stream"""
        return getattr(self.original_stream, name)


def install_global_base64_filter():
    """安装全局base64过滤器"""
    # 如果已经安装过，跳过
    if hasattr(sys.stdout, '_base64_filtered'):
        return

    # 安装过滤器
    sys.stdout = Base64Filter(sys.stdout)
    sys.stderr = Base64Filter(sys.stderr)

    # 标记已安装
    sys.stdout._base64_filtered = True
    sys.stderr._base64_filtered = True

    print("🔒 全局base64过滤器已安装")


if __name__ == "__main__":
    # 测试过滤器
    install_global_base64_filter()

    # 测试输出
    print("正常输出测试")
    print("data:image/png;base64," + "A" * 200)  # 应该被过滤
    print("长字符串测试: " + "B" * 1500)  # 应该被过滤
