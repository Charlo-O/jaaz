# ModelScope Qwen-Image 调试指南

## 🔍 问题诊断

您遇到的错误：
```
HTTP 400: {"errors":{"message":"submit image generation task error"},"request_id":"83e4559e-eabe-4544-930b-f806f61cc4b1"}
```

## 🛠️ 修复内容

我已经修复了以下问题：

### 1. **API 请求格式更新**
- ✅ 更新为 OpenAI 兼容的请求格式
- ✅ 添加了必要的 `n`, `quality`, `response_format` 参数
- ✅ 修正了参数名称（`guidance_scale` 而不是 `guidance`）

### 2. **HTTP 请求优化**
- ✅ 使用 `json=data` 而不是手动编码
- ✅ 增强的错误日志和调试信息
- ✅ 更好的 JSON 解析错误处理

### 3. **响应处理改进**
- ✅ 支持 OpenAI 兼容的响应格式 (`{"data": [{"url": "..."}]}`)
- ✅ 向后兼容旧格式 (`{"images": [{"url": "..."}]}`)

## 🚀 测试步骤

1. **重启后端服务**：
   ```bash
   cd server
   python main.py
   ```

2. **确保 ModelScope 配置正确**：
   - 检查 API Key 是否有效
   - 确认账户有足够的余额

3. **测试生成**：
   ```
   请使用 Qwen-Image 生成一张猫咪的图片
   ```

## 📊 调试信息

现在系统会输出更详细的调试信息：
```
🔥 ModelScope API request: https://api-inference.modelscope.cn/v1/images/generations
🔥 Request data: {
  "model": "Qwen/Qwen-Image",
  "prompt": "...",
  "n": 1,
  "size": "1024x1024",
  "quality": "standard",
  "response_format": "url"
}
🔥 ModelScope API response status: 200
🔥 ModelScope API response: {...}
```

## ⚠️ 可能的问题和解决方案

### 问题1：API Key 无效
**症状**：401 错误
**解决**：检查 ModelScope API Key 是否正确

### 问题2：账户余额不足
**症状**：403 错误或特定的余额错误
**解决**：为 ModelScope 账户充值

### 问题3：模型名称错误
**症状**：404 错误或模型不存在错误
**解决**：确认使用 `Qwen/Qwen-Image` 作为模型名称

### 问题4：参数格式错误
**症状**：400 错误
**解决**：我已经修复了参数格式，使用标准的 OpenAI 兼容格式

## 🔧 如果问题仍然存在

1. **检查网络连接**：确保服务器可以访问 ModelScope API
2. **验证 API Key**：在 ModelScope 官网确认 API Key 状态
3. **查看完整日志**：检查后端控制台的详细错误信息
4. **尝试其他模型**：测试 `MAILAND/majicflus_v1` 等其他 ModelScope 模型

## 📝 配置示例

确保您的 ModelScope 配置正确：
```json
{
  "modelscope": {
    "api_key": "ms-your-api-key-here",
    "url": "https://api-inference.modelscope.cn/v1/images/generations",
    "models": {
      "Qwen/Qwen-Image": {"type": "image"},
      "MAILAND/majicflus_v1": {"type": "image"}
    }
  }
}
```

## 🎯 预期结果

修复后，您应该看到：
```
🔥 ModelScope API response status: 200
🔥 ModelScope image URL: https://...
✅ 图像生成成功！
```

---

如果问题仍然存在，请提供完整的错误日志以便进一步诊断。

