import { ToolResultMessage } from '@/types/types'
import { AnimatePresence, motion } from 'motion/react'
import { Markdown } from '../Markdown'

// 过滤掉 base64 数据，避免在界面显示长串编码
const filterBase64Content = (text: string): string => {
  // 检查是否包含 base64 数据格式
  if (text.includes('data:image/') || text.includes('data:video/')) {
    // 如果内容很长且可能是 base64，显示友好提示
    if (text.length > 1000) {
      return '✅ 工具执行成功 (内容已过滤显示)'
    }
  }
  return text
}

type ToolCallContentProps = {
  expandingToolCalls: string[]
  message: ToolResultMessage
}

export const ToolCallContent: React.FC<ToolCallContentProps> = ({
  expandingToolCalls,
  message,
}) => {
  const isExpanded = expandingToolCalls.includes(message.tool_call_id)

  if (message.content.includes('<hide_in_user_ui>')) {
    return null
  }

  return (
    <AnimatePresence>
      {isExpanded && (
        <motion.div
          initial={{ opacity: 0, y: -5, height: 0 }}
          animate={{ opacity: 1, y: 0, height: 'auto' }}
          exit={{ opacity: 0, y: -5, height: 0 }}
          layout
          transition={{ duration: 0.2, ease: 'easeOut' }}
          className="p-3 bg-muted rounded-lg"
        >
          <Markdown>{filterBase64Content(message.content)}</Markdown>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

const ToolCallContentV2: React.FC<{ content: string }> = ({ content }) => {
  if (content.includes('<hide_in_user_ui>')) {
    return null
  }


  const filteredContent = filterBase64Content(content)

  return (
    <div className="p-2 bg-muted rounded-lg">
      <Markdown>{filteredContent}</Markdown>
    </div>
  )
}

export default ToolCallContentV2
