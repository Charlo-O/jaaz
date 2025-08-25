# 视频分析问题解决指南

## 问题描述
画布中显示的是base64编码的图表，而不是视频提取的关键帧。

## 问题诊断

### 第一步：确认API调用
检查前端是否调用了正确的API端点：

**✅ 正确的API**: `/api/analyze_video_to_canvas`
- 会提取视频关键帧
- 保存为PNG文件
- 添加到画布

**❌ 错误的API**: `/api/process_video`
- 只做场景分析
- 不生成关键帧图片
- 可能生成图表

### 第二步：使用诊断工具
1. 访问诊断工具：http://localhost:57988/debug_video_analysis.html
2. 按照4个步骤进行完整诊断：
   - 检查服务器状态
   - 上传测试视频
   - 测试分析API
   - 检查生成的图片

### 第三步：检查前端调用
在浏览器开发者工具中查看网络请求：
1. 打开开发者工具 (F12)
2. 切换到Network标签
3. 上传视频并分析
4. 查看实际调用的API端点

## 解决方案

### 方案1：确保使用正确的前端函数
前端应该调用：
```typescript
import { analyzeVideoAndAddToCanvas } from '@/api/upload'

// 正确的调用
const result = await analyzeVideoAndAddToCanvas(fileId, canvasId, 0.5, sessionId)
```

而不是：
```typescript
import { processVideo } from '@/api/upload'

// 错误的调用 - 这个不会生成关键帧
const result = await processVideo(file, 0.5)
```

### 方案2：清理旧的base64图片
运行清理脚本：
```bash
cd /path/to/video2
python fix_video_analysis_issue.py
```

### 方案3：手动验证API
使用curl测试正确的API：

```bash
# 1. 上传视频
curl -X POST "http://localhost:57988/api/upload_video" \
  -F "file=@your_video.mp4"

# 2. 分析视频并添加到画布 (使用上面返回的file_id)
curl -X POST "http://localhost:57988/api/analyze_video_to_canvas" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your_file_id_here",
    "canvas_id": "test-canvas-123",
    "session_id": "test-session-123",
    "threshold": 0.5
  }'
```

### 方案4：检查文件生成
查看 `server/files/` 目录，应该能找到类似这样的文件：
- `filename_keyframe_0.png`
- `filename_keyframe_1.png`
- `filename_keyframe_2.png`

## 预防措施

1. **确保前端逻辑正确**：
   - ChatTextarea.tsx 中使用 `analyzeVideoAndAddToCanvas()`
   - 不要直接调用 `processVideo()`

2. **API端点说明**：
   - `/api/analyze_video_to_canvas` - 提取关键帧并添加到画布 ✅
   - `/api/process_video` - 只做场景分析，不生成图片 ❌
   - `/api/video_analysis/{file_id}` - 获取分析结果，不生成图片 ❌

3. **验证流程**：
   - 上传视频 → 调用analyze_video_to_canvas → 检查files目录 → 验证画布

## 技术细节

### 正确的数据流
```
前端上传视频 
    ↓
/api/upload_video (返回file_id)
    ↓
/api/analyze_video_to_canvas
    ↓
TransNetV2分析 → 提取关键帧 → 保存PNG文件
    ↓
save_image_to_canvas() → 添加到画布 → WebSocket通知前端
```

### 错误的数据流 (避免)
```
前端上传视频
    ↓
/api/process_video 
    ↓
只做场景分析 → 可能生成matplotlib图表 → 返回分析结果
```

## 常见问题

**Q: 为什么会显示base64图表？**
A: 可能调用了错误的API或使用了老版本的处理逻辑，生成了matplotlib图表而不是视频关键帧。

**Q: 如何确认API调用正确？**  
A: 使用浏览器开发者工具查看网络请求，确保调用的是 `/api/analyze_video_to_canvas`。

**Q: 关键帧文件在哪里？**
A: 在 `server/files/` 目录下，文件名格式为 `{file_id}_keyframe_{index}.png`。

**Q: 如何重新分析已上传的视频？**
A: 使用视频的file_id调用 `/api/analyze_video_to_canvas` API。

## 联系支持
如果问题仍然存在，请提供：
1. 浏览器开发者工具中的网络请求截图
2. 控制台错误信息
3. `server/files/` 目录下的文件列表
4. 诊断工具的测试结果