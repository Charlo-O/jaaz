"""
Video compression utilities using FFmpeg
Optional module - only used if FFmpeg is available
"""

import os
import tempfile
import subprocess
from typing import Optional

def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def compress_video_with_ffmpeg(content: bytes, max_size_mb: float) -> Optional[bytes]:
    """
    Compress video content using FFmpeg
    
    Args:
        content: Raw video file bytes
        max_size_mb: Target maximum size in MB
    
    Returns:
        Compressed video bytes or None if compression failed
    """
    if not is_ffmpeg_available():
        print("🎬 FFmpeg not available for video compression")
        return None
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
        temp_input.write(content)
        temp_input_path = temp_input.name
    
    temp_output_path = temp_input_path.replace('.mp4', '_compressed.mp4')
    
    try:
        # Calculate target bitrate based on file size
        # Rough estimation: 1MB = 8Mbits, so target_bitrate = max_size_mb * 8 / duration_seconds
        # We'll use a conservative approach with multiple compression levels
        
        compression_attempts = [
            # Attempt 1: Medium compression (CRF 28)
            {
                'crf': '28',
                'preset': 'fast',
                'scale': 'scale=1280:-2',  # Max width 1280px, height auto (even)
                'description': 'medium compression'
            },
            # Attempt 2: Higher compression (CRF 32) 
            {
                'crf': '32',
                'preset': 'fast',
                'scale': 'scale=1280:-2',
                'description': 'high compression'
            },
            # Attempt 3: Very high compression (CRF 35) with smaller resolution
            {
                'crf': '35',
                'preset': 'fast',
                'scale': 'scale=854:-2',  # 854px width (480p equivalent)
                'description': 'very high compression'
            }
        ]
        
        for attempt in compression_attempts:
            print(f"🎬 Trying {attempt['description']} (CRF {attempt['crf']})")
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', temp_input_path,
                '-c:v', 'libx264',
                '-crf', attempt['crf'],
                '-preset', attempt['preset'],
                '-vf', attempt['scale'],
                '-c:a', 'aac',
                '-b:a', '128k',  # Audio bitrate 128k
                '-movflags', '+faststart',  # Optimize for web streaming
                '-y',  # Overwrite output file
                temp_output_path
            ]
            
            try:
                # Run FFmpeg with timeout
                result = subprocess.run(cmd, capture_output=True, 
                                      text=True, timeout=300, check=True)
                
                # Check if compressed file exists and is smaller
                if os.path.exists(temp_output_path):
                    compressed_size = os.path.getsize(temp_output_path)
                    compressed_size_mb = compressed_size / (1024 * 1024)
                    
                    print(f"🎬 Compressed to {compressed_size_mb:.2f}MB with {attempt['description']}")
                    
                    # If within target size, use this version
                    if compressed_size_mb <= max_size_mb:
                        with open(temp_output_path, 'rb') as f:
                            compressed_content = f.read()
                        
                        print(f"🎬 Video compression successful: {len(content)/(1024*1024):.2f}MB → {compressed_size_mb:.2f}MB")
                        return compressed_content
                    else:
                        print(f"🎬 Still too large ({compressed_size_mb:.2f}MB > {max_size_mb}MB), trying next level")
                        
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"🎬 FFmpeg compression failed for {attempt['description']}: {e}")
                continue
            
            finally:
                # Clean up intermediate files
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
        
        print("🎬 All compression attempts failed or exceeded size limit")
        return None
        
    finally:
        # Clean up input temp file
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)

def get_video_info_ffprobe(file_path: str) -> dict:
    """
    Get video information using FFprobe
    
    Returns:
        Dict with video metadata or default values if FFprobe unavailable
    """
    if not is_ffmpeg_available():
        return {
            'width': 1920,
            'height': 1080,
            'duration': 0,
            'fps': 30.0
        }
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              timeout=10, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream:
            width = int(video_stream.get('width', 1920))
            height = int(video_stream.get('height', 1080))
            
            # Calculate FPS
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Get duration
            duration = float(data.get('format', {}).get('duration', 0))
            
            return {
                'width': width,
                'height': height,
                'duration': round(duration, 2),
                'fps': round(fps, 2)
            }
    
    except Exception as e:
        print(f"🎬 FFprobe failed: {e}")
    
    # Return defaults if FFprobe fails
    return {
        'width': 1920,
        'height': 1080,
        'duration': 0,
        'fps': 30.0
    }