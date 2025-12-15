import { compressImageFile } from '@/utils/imageUtils'

export async function uploadImage(
  file: File
): Promise<{ file_id: string; width: number; height: number; url: string }> {
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

export interface VideoKeyframe {
  file_id: string
  url: string
  width: number
  height: number
  frame_index: number
  timestamp: number
}

export interface VideoAnalysisResult {
  success: boolean
  keyframes: VideoKeyframe[]
  total: number
  mode: string
  video_id?: string
  video_url?: string
  warning?: string
}

export async function uploadVideo(
  file: File
): Promise<{ video_id: string; url: string; filename: string }> {
  const formData = new FormData()
  formData.append('file', file)
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

export async function analyzeVideo(
  videoId: string,
  options: {
    mode?: 'transnet' | 'simple'
    threshold?: number
    numFrames?: number
    minSceneLength?: number
  } = {}
): Promise<VideoAnalysisResult> {
  const formData = new FormData()
  formData.append('video_id', videoId)
  formData.append('mode', options.mode || 'simple')
  formData.append('threshold', String(options.threshold || 0.5))
  formData.append('num_frames', String(options.numFrames || 10))
  formData.append('min_scene_length', String(options.minSceneLength || 10))

  const response = await fetch('/api/video/analyze', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to analyze video')
  }
  return await response.json()
}

export async function extractKeyframesFromVideo(
  file: File,
  options: {
    mode?: 'transnet' | 'simple'
    threshold?: number
    numFrames?: number
    minSceneLength?: number
  } = {}
): Promise<VideoAnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('mode', options.mode || 'simple')
  formData.append('threshold', String(options.threshold || 0.5))
  formData.append('num_frames', String(options.numFrames || 10))
  formData.append('min_scene_length', String(options.minSceneLength || 10))

  const response = await fetch('/api/video/extract_keyframes', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to extract keyframes')
  }
  return await response.json()
}
