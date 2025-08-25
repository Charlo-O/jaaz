# 视频AI集成使用文档

## 概述

本项目现在完全支持将视频文件发送给大模型进行分析，就像图片一样。支持多种AI模型，包括OpenAI GPT-4o和Google Gemini。

## 支持的AI模型

### OpenAI
- **图片支持**: GPT-4 Vision, GPT-4o, GPT-4o-mini
- **视频支持**: GPT-4o, GPT-4o-mini ✅
- **配置**: 需要在设置中配置OpenAI API密钥

### Google Gemini  
- **图片支持**: Gemini Pro Vision, Gemini 1.5 Pro, Gemini 1.5 Flash
- **视频支持**: Gemini 1.5 Pro, Gemini 1.5 Flash ✅
- **配置**: 需要在设置中配置Google AI API密钥

### Claude (Anthropic)
- **图片支持**: Claude 3.5 Sonnet, Claude 3 Haiku ✅
- **视频支持**: 暂不支持 ❌

## 使用流程

### 1. 上传视频
```
用户操作 → 选择视频文件 → 前端压缩(>50MB) → 上传到服务器 → 后端压缩(>200MB) → 存储 + 生成缩略图
```

### 2. 视频分析
```
发送消息 → 检测视频内容 → 转换为base64 → 调用AI模型API → 返回分析结果
```

### 3. 前端显示
- 聊天输入框显示视频缩略图和时长
- 支持删除和重新上传
- AI返回详细的视频内容分析

## API使用

### 直接调用API
```bash
curl -X POST "http://localhost:57988/api/analyze_media" \
  -H "Content-Type: application/json" \
  -d '{
    "media_content": "data:video/mp4;base64,UklGRmhI...",
    "media_type": "video", 
    "prompt": "请描述这个视频的内容",
    "provider": "openai",
    "model": "gpt-4o"
  }'
```

### 查看支持的模型
```bash
curl "http://localhost:57988/api/supported_models"
```

## 消息格式

### 前端发送格式
```typescript
{
  role: 'user',
  content: [
    { type: 'text', text: '请分析这个视频' },
    { type: 'video_url', video_url: { url: 'data:video/mp4;base64,UklGR...' }}
  ]
}
```

### 后端处理格式
```python
# 检测消息类型
for content_item in user_message['content']:
    if content_item.get('type') == 'video_url':
        video_content = content_item.get('video_url', {}).get('url', "")
        # 调用多模态AI服务分析
```

## 配置要求

### OpenAI配置
```json
{
  "openai": {
    "api_key": "sk-your-api-key",
    "url": "https://api.openai.com/v1"
  }
}
```

### Google配置  
```json
{
  "google": {
    "api_key": "your-google-api-key", 
    "url": "https://generativelanguage.googleapis.com"
  }
}
```

## 视频格式支持

### 上传支持
- **格式**: MP4, AVI, MOV, MKV, WMV, FLV, WebM, 3GP, OGV
- **大小**: 最大300MB (压缩后)
- **压缩**: 自动压缩大文件

### AI模型支持
- **OpenAI**: 主要支持MP4格式
- **Google**: 支持多种格式，自动检测MIME类型
- **格式转换**: 必要时自动转换为模型支持的格式

## 使用示例

### 1. 视频内容分析
**用户**: [上传视频] "请描述这个视频中发生了什么"

**AI回复**: 
```
✨ VIDEO分析结果 (由 openai/gpt-4o 提供):

这个视频显示了一个阳光明媚的公园场景。视频开始时，一个小女孩正在草地上踢足球。她穿着红色的运动衫和蓝色的短裤。背景中可以看到绿树成荫的小径和几个正在散步的人。

大约在视频的中段，小女孩踢球时球滚向了一只正在草地上玩耍的金毛犬。狗狗很兴奋地开始追逐足球，摇着尾巴跑来跑去。小女孩笑着跟着狗狗一起跑。

视频的最后部分显示了女孩的父母从远处走过来，他们微笑着看着这个温馨的场面。整个视频给人一种轻松愉快的家庭时光的感觉。
```

### 2. 视频技术分析
**用户**: [上传视频] "分析这个视频的拍摄技巧和画面质量"

**AI回复**:
```
✨ VIDEO分析结果 (由 gemini-1.5-pro 提供):

从技术角度分析这个视频：

拍摄技巧：
- 使用了稳定的手持拍摄，画面抖动很少
- 采用了中景到特写的多种镜头语言
- 光线运用得当，利用自然光营造温暖氛围
- 构图遵循三分法则，主体位置安排合理

画面质量：
- 分辨率：1920x1080 (Full HD)
- 帧率：30fps，动作流畅
- 色彩饱和度适中，色温偏暖
- 对焦准确，景深控制得当
- 轻微的运动模糊在快速动作场景中是正常的

整体评价：这是一个技术水准较高的家庭录像，展现了良好的拍摄基础。
```

## 错误处理

### 常见错误
1. **模型不支持视频**: 自动切换到支持视频的模型
2. **API密钥未配置**: 返回配置提示信息  
3. **视频文件过大**: 自动压缩或提示文件过大
4. **网络超时**: 重试机制和超时提示
5. **格式不支持**: 提示支持的格式列表

### 错误响应示例
```json
{
  "success": false,
  "error": "Model claude-3-5-sonnet-20241022 does not support video processing. Please use OpenAI GPT-4o or Google Gemini 1.5 Pro.",
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022"
}
```

## 性能优化

### 前端优化
- 视频预压缩减少传输时间
- 缩略图预览提升用户体验
- 异步上传不阻塞界面

### 后端优化  
- 并行处理视频元数据提取
- 智能选择最适合的AI模型
- 缓存常用分析结果

### AI调用优化
- 自动重试失败的请求
- 智能降级到可用模型
- 合理的超时设置

## 安全考虑

- 视频文件大小限制防止滥用
- API密钥加密存储
- 用户上传内容不会被永久保存
- 遵循各AI服务商的使用条款

这个视频AI集成功能为用户提供了强大的视频内容理解能力，支持从简单的内容描述到复杂的技术分析等多种用途。