import { compressImageFile } from '@/utils/imageUtils'
import { compressVideoFile } from '@/utils/videoUtils'

export interface UploadImageResponse {
  file_id: string
  width: number
  height: number
  url: string
}

export interface UploadVideoResponse {
  file_id: string
  url: string
  thumbnail_url?: string
  width: number
  height: number
  duration: number
  fps: number
  size_mb: number
  type: 'video'
}

export interface VideoProcessResponse {
  file_id: string
  url: string
  thumbnail_url?: string
  type: 'video'
  analysis: {
    video_path: string
    video_info: {
      fps: number
      total_frames: number
      duration: number
    }
    scene_detection: {
      method: string
      threshold: number
      total_scenes: number
      scenes: Array<{
        scene_id: number
        start_frame: number
        end_frame: number
        start_time: number
        end_time: number
        duration: number
      }>
    }
    predictions: number[]
  }
  message: string
}

export async function uploadImage(file: File): Promise<UploadImageResponse> {
  // Compress image before upload
  const compressedFile = await compressImageFile(file)

  const formData = new FormData()
  formData.append('file', compressedFile)
  const response = await fetch('/api/upload_image', {
    method: 'POST',
    body: formData,
  })
  return await response.json()
}

export async function uploadVideo(file: File): Promise<UploadVideoResponse> {
  // Compress video before upload if it's large
  const compressedFile = await compressVideoFile(file)

  const formData = new FormData()
  formData.append('file', compressedFile)
  const response = await fetch('/api/upload_video', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to upload video')
  }

  return await response.json()
}

export async function processVideo(file: File, threshold: number = 0.5): Promise<VideoProcessResponse> {
  // Compress video before upload if it's large
  const compressedFile = await compressVideoFile(file)

  const formData = new FormData()
  formData.append('file', compressedFile)
  formData.append('threshold', threshold.toString())

  const response = await fetch('/api/process_video', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to process video')
  }

  return await response.json()
}

<<<<<<< Updated upstream
export async function analyzeVideoAndAddToCanvas(fileId: string, canvasId: string, threshold: number = 0.5, sessionId?: string): Promise<{ success: boolean, message: string, analysis?: any }> {
=======
export interface VideoAnalysisResponse {
  success: boolean
  message: string
  analysis?: any
  key_frames?: Array<{
    filename: string
    url: string
    scene_index: number
    width: number
    height: number
  }>
  total_scenes?: number
  total_key_frames?: number
}

export async function analyzeVideoAndAddToCanvas(fileId: string, canvasId: string, threshold: number = 0.5, sessionId?: string): Promise<VideoAnalysisResponse> {
>>>>>>> Stashed changes
  const response = await fetch('/api/analyze_video_to_canvas', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_id: fileId,
      canvas_id: canvasId,
      session_id: sessionId || canvasId, // 使用sessionId或降级到canvasId
      threshold: threshold,
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to analyze video')
  }

  return await response.json()
}

export async function uploadFile(file: File): Promise<UploadImageResponse | UploadVideoResponse> {
  // Get file type from MIME type first, fallback to extension
  let fileType = file.type.split('/')[0]

  // If MIME type is not available or generic, check file extension
  if (!fileType || fileType === 'application' || fileType === 'octet-stream') {
    const extension = file.name.split('.').pop()?.toLowerCase()

    const imageExts = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'svg', 'ico']
    const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', '3gp', 'ogv']

    if (extension && imageExts.includes(extension)) {
      fileType = 'image'
    } else if (extension && videoExts.includes(extension)) {
      fileType = 'video'
    }
  }

  console.log(`🔍 File type detection: ${file.name}, MIME: ${file.type}, detected type: ${fileType}`)

  if (fileType === 'image') {
    return uploadImage(file)
  } else if (fileType === 'video') {
    return uploadVideo(file)
  } else {
    throw new Error(`Unsupported file type: ${fileType}. Only images and videos are supported.`)
  }
}
