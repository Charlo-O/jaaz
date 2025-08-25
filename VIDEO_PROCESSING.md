# 视频处理流程文档

本项目现在支持与图片相同的双重压缩逻辑来处理视频文件。

## 1. 前端视频压缩流程

### 1.1 压缩触发条件
```typescript
// react/src/utils/videoUtils.ts - compressVideoFile()
export async function compressVideoFile(file: File): Promise<File> {
    const fileSizeMB = file.size / (1024 * 1024)
    const maxSizeMB = 50  // 50MB前端压缩阈值
    
    // 只有超过50MB的视频才会被压缩
    if (fileSizeMB <= maxSizeMB) {
        return file  // 小视频直接返回原文件
    }
    
    // 大视频进行压缩处理
    const compressedFile = await compressLargeVideo(file)
    return compressedFile
}
```

### 1.2 前端压缩策略
- **阈值**: 50MB
- **方法**: MediaRecorder API + Canvas重绘
- **目标码率**: 2Mbps
- **最大分辨率**: 1920px宽度
- **帧率**: 30fps
- **输出格式**: WebM (VP9编码)

### 1.3 前端压缩实现
```typescript
function compressLargeVideo(file: File): Promise<File> {
    // 1. 加载视频到<video>元素
    // 2. 获取原始尺寸，按比例缩放到最大1920px
    // 3. 使用Canvas.captureStream()创建压缩流
    // 4. MediaRecorder录制压缩视频
    // 5. 返回WebM格式压缩文件
}
```

## 2. 后端视频压缩流程

### 2.1 压缩触发条件
```python
# server/routers/video_router.py
@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...), max_size_mb: float = 200.0):
    original_size_mb = len(content) / (1024 * 1024)
    
    # 超过200MB触发后端压缩
    needs_compression = original_size_mb > max_size_mb
```

### 2.2 后端压缩策略
- **第一阈值**: 200MB (触发压缩)
- **最大允许**: 300MB (压缩失败时的上限)
- **压缩工具**: FFmpeg (可选)
- **压缩级别**: 多级压缩尝试

### 2.3 FFmpeg压缩实现
```python
# server/utils/video_compression.py
def compress_video_with_ffmpeg(content: bytes, max_size_mb: float):
    compression_attempts = [
        # 尝试1: 中等压缩 (CRF 28, 1280px)
        {
            'crf': '28',
            'preset': 'fast', 
            'scale': 'scale=1280:-2',
            'description': 'medium compression'
        },
        # 尝试2: 高压缩 (CRF 32, 1280px)
        {
            'crf': '32',
            'preset': 'fast',
            'scale': 'scale=1280:-2', 
            'description': 'high compression'
        },
        # 尝试3: 极高压缩 (CRF 35, 854px)
        {
            'crf': '35',
            'preset': 'fast',
            'scale': 'scale=854:-2',
            'description': 'very high compression'
        }
    ]
```

## 3. 完整处理链路

```
用户选择视频文件
       ↓
前端检查大小 (>50MB?)
       ↓ 是
Canvas + MediaRecorder压缩 (1920px, 2Mbps, WebM)
       ↓
上传到后端
       ↓
后端检查大小 (>200MB?)
       ↓ 是
FFmpeg多级压缩:
- CRF 28 (1280px) → CRF 32 → CRF 35 (854px)
- 音频128k AAC
- H.264编码MP4输出
       ↓
检查是否 ≤200MB 或 ≤300MB (fallback)
       ↓
保存到FILES_DIR
       ↓
提取元数据 (FFprobe > OpenCV)
       ↓
生成缩略图 (第一帧)
       ↓
返回视频信息
```

## 4. 元数据提取

### 4.1 优先级
1. **FFprobe** (FFmpeg工具) - 最准确
2. **OpenCV** - 备用方案
3. **默认值** - 最后fallback

### 4.2 提取信息
```python
video_info = {
    'width': int,      # 视频宽度
    'height': int,     # 视频高度  
    'duration': float, # 时长(秒)
    'fps': float       # 帧率
}
```

## 5. 缩略图生成

```python
def generate_video_thumbnail(file_path: str, file_id: str):
    # 1. OpenCV读取视频第一帧
    # 2. BGR转RGB颜色空间
    # 3. PIL调整大小 (最大300x300)
    # 4. 保存为JPEG格式 (85%质量)
    # 5. 文件名: {file_id}_thumb.jpg
```

## 6. 存储结构

```
FILES_DIR/
├── abc123.mp4              # 原始或压缩后视频
├── abc123_thumb.jpg        # 视频缩略图
├── def456.webm             # 前端压缩的WebM视频  
├── def456_thumb.jpg        # 对应缩略图
├── ghi789.mp4              # FFmpeg压缩的MP4视频
└── ghi789_thumb.jpg        # 对应缩略图
```

## 7. 前端显示逻辑

### 7.1 预览显示
```tsx
{item.type === 'video' ? (
    <>
        {/* 显示缩略图 */}
        <img src={item.thumbnail_url} alt="Video thumbnail" />
        
        {/* 视频时长标识 */}
        <div className="duration-badge">
            {Math.round(item.duration)}s
        </div>
    </>
) : (
    <img src={`/api/file/${item.file_id}`} alt="Uploaded image" />
)}
```

### 7.2 消息内容生成
```typescript
// 视频和图片分别处理
if (videoFiles.length > 0) {
    text_content += `\n\n<input_videos count="${videoFiles.length}">`
    videoFiles.forEach((video, index) => {
        text_content += `\n<video index="${index + 1}" file_id="${video.file_id}" width="${video.width}" height="${video.height}" duration="${video.duration}" />`
    })
    text_content += `\n</input_videos>`
}
```

## 8. 压缩效果对比

| 场景 | 原始大小 | 前端压缩 | 后端压缩 | 最终大小 |
|------|----------|----------|----------|----------|
| 小视频 | 20MB | 无压缩 | 无压缩 | 20MB |
| 中等视频 | 100MB | 压缩到40MB | 无压缩 | 40MB |
| 大视频 | 500MB | 压缩到45MB | 压缩到150MB | 150MB |
| 超大视频 | 1GB | 压缩到48MB | 压缩到180MB | 180MB |

## 9. 错误处理

- **前端压缩失败**: 使用原文件上传
- **后端压缩失败**: 允许300MB以内原文件
- **FFmpeg不可用**: 自动降级到基础处理
- **元数据提取失败**: 使用默认值
- **缩略图生成失败**: 返回空值，前端显示默认图标

## 10. 优化建议

### 10.1 生产环境
- 安装FFmpeg二进制文件
- 考虑使用专业视频处理服务(AWS MediaConvert等)
- 实现异步压缩处理
- 添加进度显示

### 10.2 性能优化
- 视频压缩放到后台队列
- 使用更高效的编码器
- 根据视频内容智能调整参数
- 缓存常见分辨率的预设

这个双重压缩机制确保了视频文件在传输和存储时都保持合理的大小，同时尽可能保持视觉质量。