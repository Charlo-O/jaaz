/**
 * Video processing utilities
 */

interface ProcessedVideo {
  url: string
  filename: string
  duration?: number
  width?: number
  height?: number
}

/**
 * Get video metadata
 */
function getVideoMetadata(file: File): Promise<{
  duration: number
  width: number
  height: number
}> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    
    video.onloadedmetadata = () => {
      resolve({
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight
      })
      URL.revokeObjectURL(video.src)
    }
    
    video.onerror = () => {
      reject(new Error(`Failed to load video metadata: ${file.name}`))
      URL.revokeObjectURL(video.src)
    }
    
    video.src = URL.createObjectURL(file)
  })
}

/**
 * Compress large video using MediaRecorder API
 */
function compressLargeVideo(file: File): Promise<File> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    
    video.onloadedmetadata = async () => {
      try {
        // Calculate new dimensions (max 1920px width)
        let { videoWidth: width, videoHeight: height } = video
        const maxWidth = 1920
        
        if (width > maxWidth) {
          const ratio = maxWidth / width
          width = Math.round(width * ratio)
          height = Math.round(height * ratio)
        }
        
        canvas.width = width
        canvas.height = height
        
        // Create MediaRecorder for compression
        const stream = canvas.captureStream(30) // 30fps
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'video/webm;codecs=vp9', // Try VP9 first
          videoBitsPerSecond: 2000000 // 2Mbps target bitrate
        })
        
        const chunks: Blob[] = []
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data)
          }
        }
        
        mediaRecorder.onstop = () => {
          const compressedBlob = new Blob(chunks, { type: 'video/webm' })
          const compressedFile = new File([compressedBlob], 
            file.name.replace(/\.[^/.]+$/, '.webm'), 
            { type: 'video/webm' }
          )
          
          console.log(
            `Video compressed: ${file.name} (${Math.round(file.size / 1024 / 1024)}MB → ${Math.round(compressedFile.size / 1024 / 1024)}MB)`
          )
          
          resolve(compressedFile)
        }
        
        mediaRecorder.onerror = (error) => {
          reject(new Error(`Video compression failed: ${error}`))
        }
        
        // Start recording
        mediaRecorder.start()
        
        // Play video and draw frames to canvas
        video.currentTime = 0
        video.play()
        
        const drawFrame = () => {
          if (!video.paused && !video.ended) {
            ctx?.drawImage(video, 0, 0, width, height)
            requestAnimationFrame(drawFrame)
          } else {
            // Video finished, stop recording
            setTimeout(() => {
              mediaRecorder.stop()
              stream.getTracks().forEach(track => track.stop())
            }, 100)
          }
        }
        
        video.onplaying = drawFrame
        
      } catch (error) {
        reject(new Error(`Failed to compress video: ${file.name}`))
      } finally {
        URL.revokeObjectURL(video.src)
      }
    }
    
    video.onerror = () => {
      reject(new Error(`Failed to load video: ${file.name}`))
      URL.revokeObjectURL(video.src)
    }
    
    video.src = URL.createObjectURL(file)
  })
}

/**
 * Simple video compression using lower bitrate re-encoding
 */
function simpleVideoCompress(file: File): Promise<File> {
  return new Promise((resolve, reject) => {
    // For now, just return original file as browser video compression is complex
    // In production, you'd want to use FFmpeg.wasm or server-side compression
    console.log(`Video compression not fully implemented, returning original: ${file.name}`)
    resolve(file)
  })
}

/**
 * Compress video file and return compressed File object
 */
export async function compressVideoFile(file: File): Promise<File> {
  // Check file size (50MB = 50 * 1024KB)
  const fileSizeMB = file.size / (1024 * 1024)
  const maxSizeMB = 50
  
  // If file is small enough, return as is
  if (fileSizeMB <= maxSizeMB) {
    return file
  }
  
  console.log(
    `Compressing large video: ${file.name} (${Math.round(fileSizeMB)}MB)`
  )
  
  try {
    // Try simple compression first
    const compressedFile = await simpleVideoCompress(file)
    
    // If still too large, try more aggressive compression
    if (compressedFile.size / (1024 * 1024) > maxSizeMB) {
      console.log(`Attempting advanced compression for: ${file.name}`)
      return await compressLargeVideo(file)
    }
    
    return compressedFile
  } catch (error) {
    console.warn(
      `Failed to compress video ${file.name}, using original:`,
      error
    )
    return file
  }
}

/**
 * Process video files - compress only if larger than 50MB
 */
export async function processVideoFiles(
  files: File[]
): Promise<ProcessedVideo[]> {
  const results = await Promise.allSettled(
    files.map(async (file) => {
      // Get video metadata
      let metadata
      try {
        metadata = await getVideoMetadata(file)
      } catch (error) {
        console.warn(`Failed to get metadata for ${file.name}:`, error)
        metadata = { duration: 0, width: 1920, height: 1080 }
      }
      
      // Check file size (50MB)
      const fileSizeMB = file.size / (1024 * 1024)
      
      let processedFile: File
      if (fileSizeMB > 50) {
        // Large file - compress it
        console.log(
          `[Silent] Compressing large video: ${file.name} (${Math.round(fileSizeMB)}MB)`
        )
        processedFile = await compressVideoFile(file)
      } else {
        // Small file - use as is
        processedFile = file
      }
      
      return {
        url: URL.createObjectURL(processedFile),
        filename: processedFile.name,
        duration: metadata.duration,
        width: metadata.width,
        height: metadata.height,
      }
    })
  )
  
  // Extract successful results
  const processedVideos: ProcessedVideo[] = []
  const errors: string[] = []
  
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      processedVideos.push(result.value)
    } else {
      errors.push(`${files[index].name}: ${result.reason.message}`)
    }
  })
  
  // Handle errors
  if (errors.length > 0 && processedVideos.length === 0) {
    throw new Error(`All videos failed to process:\n${errors.join('\n')}`)
  }
  
  if (errors.length > 0) {
    console.warn('Some videos failed to process:', errors)
  }
  
  return processedVideos
}

/**
 * Get video file duration without full processing
 */
export function getVideoDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    
    video.onloadedmetadata = () => {
      resolve(video.duration)
      URL.revokeObjectURL(video.src)
    }
    
    video.onerror = () => {
      reject(new Error(`Failed to get video duration: ${file.name}`))
      URL.revokeObjectURL(video.src)
    }
    
    video.src = URL.createObjectURL(file)
  })
}