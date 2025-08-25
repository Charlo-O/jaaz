#!/usr/bin/env python3
"""
测试TransNetV2模型加载
"""
import os
import sys


def test_transnetv2_pytorch():
    """测试PyTorch版本的TransNetV2"""
    print("\n🔍 Testing PyTorch TransNetV2...")

    try:
        # 添加TransNetV2路径
        pytorch_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "TransNetV2",
            "inference-pytorch",
        )
        sys.path.insert(0, pytorch_path)

        # 检查权重文件
        weights_path = os.path.join(
            pytorch_path, "transnetv2-pytorch-weights", "transnetv2-pytorch-weights.pth"
        )
        if not os.path.exists(weights_path):
            print(f"❌ PyTorch权重文件不存在: {weights_path}")
            print("💡 需要运行权重转换脚本")
            return False

        print(f"✅ PyTorch权重文件存在: {weights_path}")

        # 测试导入PyTorch
        try:
            import torch

            print(f"✅ PyTorch版本: {torch.__version__}")
        except ImportError as e:
            print(f"❌ PyTorch导入失败: {e}")
            return False

        # 测试加载TransNetV2
        try:
            from transnetv2_pytorch import TransNetV2

            model = TransNetV2()
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            print("✅ TransNetV2 PyTorch模型加载成功!")
            return True
        except Exception as e:
            print(f"❌ TransNetV2 PyTorch模型加载失败: {e}")
            return False

    except Exception as e:
        print(f"❌ PyTorch版本测试失败: {e}")
        return False


def convert_pytorch_weights():
    """转换PyTorch权重文件"""
    print("\n🔄 Converting PyTorch weights...")

    try:
        pytorch_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "TransNetV2",
            "inference-pytorch",
        )
        os.chdir(pytorch_path)

        # 运行转换脚本
        import subprocess

        result = subprocess.run(
            [sys.executable, "convert_weights.py"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✅ 权重转换成功!")
            print(result.stdout)
            return True
        else:
            print("❌ 权重转换失败!")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ 权重转换异常: {e}")
        return False


def test_dependencies():
    """测试依赖包"""
    print("🔍 Testing dependencies...")

    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('ffmpeg', 'ffmpeg-python'),
    ]

    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            if module == 'torch':
                import torch

                print(f"✅ {name}: {torch.__version__}")
            elif module == 'numpy':
                import numpy as np

                print(f"✅ {name}: {np.__version__}")
            elif module == 'cv2':
                import cv2

                print(f"✅ {name}: {cv2.__version__}")
            else:
                print(f"✅ {name}: installed")
        except ImportError:
            print(f"❌ {name}: not installed")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("TransNetV2 模型加载测试")
    print("=" * 60)

    # 测试依赖
    deps_ok = test_dependencies()

    if not deps_ok:
        print("\n❌ 依赖包不完整，请先安装所需依赖")
        sys.exit(1)

    # 测试PyTorch版本
    pytorch_ok = test_transnetv2_pytorch()

    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"PyTorch版本: {'✅ 成功' if pytorch_ok else '❌ 失败'}")

    if pytorch_ok:
        print("🎉 TransNetV2服务可以正常工作!")
    else:
        print("❌ PyTorch版本失败，需要检查模型文件和依赖")

    print("=" * 60)
