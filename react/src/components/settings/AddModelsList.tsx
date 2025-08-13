import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Plus, Trash2 } from 'lucide-react'
import { Dialog, DialogContent, DialogTrigger } from '../ui/dialog'

export type ModelItem = {
  name: string
  type: 'text' | 'image' | 'video'
}

interface ModelsListProps {
  models: Record<string, { type?: 'text' | 'image' | 'video' }>
  onChange: (
    models: Record<string, { type?: 'text' | 'image' | 'video' }>
  ) => void
  label?: string
  // 强制所有新增模型为某一类型（例如 ModelScope 仅支持 image）
  forceType?: 'text' | 'image' | 'video'
  // 输入框联想候选
  suggestions?: string[]
  // 自定义占位符
  placeholder?: string
}

export default function AddModelsList({
  models,
  onChange,
  label = 'Models',
  forceType,
  suggestions = [],
  placeholder,
}: ModelsListProps) {
  const [modelItems, setModelItems] = useState<ModelItem[]>([])
  const [newModelName, setNewModelName] = useState('')
  const [openAddModelDialog, setOpenAddModelDialog] = useState(false)
  const [isOpenSuggest, setIsOpenSuggest] = useState(false)

  useEffect(() => {
    const items = Object.entries(models).map(([name, config]) => ({
      name,
      // 如果传入 forceType，则以该类型覆盖
      type: (forceType || config.type || 'text') as 'text' | 'image' | 'video',
    }))
    setModelItems(items.length > 0 ? items : [])
  }, [models, forceType])

  const notifyChange = useCallback(
    (items: ModelItem[]) => {
      // Filter out empty model names and convert back to object format
      const validModels = items.filter((model) => model.name.trim())
      const modelsConfig: Record<
        string,
        { type?: 'text' | 'image' | 'video' }
      > = {}

      validModels.forEach((model) => {
        modelsConfig[model.name] = { type: model.type }
      })

      onChange(modelsConfig)
    },
    [onChange]
  )

  const handleAddModel = () => {
    if (newModelName) {
      const newItems = [
        ...modelItems,
        { name: newModelName, type: (forceType || 'text') as 'text' | 'image' | 'video' },
      ]
      setModelItems(newItems)
      notifyChange(newItems)
      setNewModelName('')
      setOpenAddModelDialog(false)
      setIsOpenSuggest(false)
    }
  }

  const handleRemoveModel = (index: number) => {
    if (modelItems.length > 1) {
      const newItems = modelItems.filter((_, i) => i !== index)
      setModelItems(newItems)
      notifyChange(newItems)
    }
  }
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <Dialog open={openAddModelDialog} onOpenChange={setOpenAddModelDialog}>
          <DialogTrigger asChild>
            <Button variant="secondary" size="sm">
              <Plus className="h-4 w-4 mr-1" />
              Add Model
            </Button>
          </DialogTrigger>
          <DialogContent>
            <div className="space-y-5">
              <Label>Model Name</Label>
              <Input
                type="text"
                placeholder={placeholder || 'provider/model-name'}
                value={newModelName}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleAddModel()
                  }
                }}
                onFocus={() => setIsOpenSuggest(true)}
                onBlur={() => setTimeout(() => setIsOpenSuggest(false), 120)}
                onChange={(e) => {
                  setNewModelName(e.target.value)
                  setIsOpenSuggest(true)
                }}
              />
              {isOpenSuggest && suggestions.length > 0 && (
                <div className="max-h-60 overflow-auto rounded-md border bg-popover shadow p-1">
                  {suggestions
                    .filter((s) =>
                      s.toLowerCase().includes(newModelName.toLowerCase())
                    )
                    .slice(0, 50)
                    .map((s) => (
                      <div
                        key={s}
                        className="cursor-pointer rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => {
                          setNewModelName(s)
                          setIsOpenSuggest(false)
                        }}
                      >
                        {s}
                      </div>
                    ))}
                </div>
              )}
              <Button type="button" onClick={handleAddModel} className="w-full">
                Add Model
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-2">
        {modelItems.map((model, index) => (
          <div key={index} className="flex items-center justify-between">
            <p className="w-[50%]">{model.name}</p>
            <div className="flex items-center gap-6">
              <p>{model.type}</p>
              {modelItems.length > 1 && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => handleRemoveModel(index)}
                  className="h-10 w-10 p-0"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
