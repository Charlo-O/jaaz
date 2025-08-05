# Jaaz - Enhanced Midjourney Integration

This is an enhanced version of Jaaz with improved Midjourney upscale functionality.

## 🆕 Midjourney Enhancement

### New Upscale Feature
- **Automatic Upscaling**: Generate 4-grid image and automatically upscale all 4 images
- **Optimized Timing**: Submit upscale requests with 5-second intervals (not waiting for completion)
- **Parallel Processing**: All 4 upscale tasks run simultaneously on Midjourney servers
- **Configurable**: Can disable upscale to get only the 4-grid image

### Usage
```python
# Auto upscale all 4 images (default behavior)
await generate_image_by_midjourney(
    prompt="a beautiful cat",
    aspect_ratio="1:1",
    enable_upscale=True  # Default
)

# Only generate 4-grid image
await generate_image_by_midjourney(
    prompt="a beautiful cat", 
    aspect_ratio="1:1",
    enable_upscale=False
)
```

### Implementation Details
1. **Step 1**: Generate 4-grid image using `/mj/submit/imagine`
2. **Step 2**: Submit 4 upscale requests with 5s intervals using `/mj/submit/change`
3. **Step 3**: Collect all 4 upscaled images in parallel

### Files Modified
- `server/tools/image_providers/midjourney_provider.py` - Added upscale functionality
- `server/tools/generate_image_by_midjourney.py` - Added enable_upscale parameter
- `server/tools/utils/image_generation_core.py` - Added upscale support

## 📋 Installation

### Prerequisites
- Python >= 3.12
- Node.js
- Midjourney Proxy API server

### Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   # Frontend
   cd react
   npm install --force
   npm run build
   
   # Backend
   cd ../server
   pip install -r requirements.txt
   ```

3. Configure Midjourney API in settings
4. Start the application:
   ```bash
   python server/main.py
   ```

## 🔧 Development

```bash
# Frontend development
cd react
npm run dev

# Backend development
cd server
python main.py
```

## 📄 Original README

For the original Jaaz documentation, see:
- [README.md](./README.md) - English version
- [README_zh.md](./README_zh.md) - Chinese version

## 🛠️ Configuration

Make sure to configure your Midjourney Proxy API settings in the application settings:
- API URL: Your Midjourney proxy server URL
- API Key: Your API key (if required)

## 📝 License

Same as original Jaaz project.