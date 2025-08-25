#!/usr/bin/env python3
"""
直接测试TransNetV2服务
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("=" * 60)
    print("测试TransNetV2服务初始化")
    print("=" * 60)
    
    try:
        from services.transnetv2_service import TransNetV2Service
        
        print("🔍 创建TransNetV2Service实例...")
        service = TransNetV2Service()
        
        print(f"\n📊 服务状态:")
        print(f"模型类型: {service.model_type}")
        print(f"模型对象: {service.model}")
        
        if service.model is not None:
            print("✅ TransNetV2服务初始化成功!")
        else:
            print("⚠️ TransNetV2模型未加载，将使用基本检测方法")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)