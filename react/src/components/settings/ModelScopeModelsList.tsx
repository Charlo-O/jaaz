import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Plus, Trash2, PencilIcon } from 'lucide-react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../ui/dialog'
import { useTranslation } from 'react-i18next'

export type ModelScopeModelItem = {
    name: string
    type: 'text' | 'image' | 'video'
    description?: string
}

interface ModelScopeModelsListProps {
    models: Record<string, { type?: 'text' | 'image' | 'video'; description?: string }>
    onChange: (
        models: Record<string, { type?: 'text' | 'image' | 'video'; description?: string }>
    ) => void
    label?: string
}

export default function ModelScopeModelsList({
    models,
    onChange,
    label = 'ModelScope Models',
}: ModelScopeModelsListProps) {
    const { t } = useTranslation()
    const [modelItems, setModelItems] = useState<ModelScopeModelItem[]>([])
    const [newModelName, setNewModelName] = useState('')
    const [newModelDescription, setNewModelDescription] = useState('')
    const [editingModel, setEditingModel] = useState<ModelScopeModelItem | null>(null)
    const [openAddModelDialog, setOpenAddModelDialog] = useState(false)

    useEffect(() => {
        const modelItems = Object.entries(models).map(([name, config]) => ({
            name,
            type: (config.type || 'image') as 'text' | 'image' | 'video',
            description: config.description || '',
        }))
        setModelItems(modelItems.length > 0 ? modelItems : [])
    }, [models])

    const notifyChange = useCallback(
        (items: ModelScopeModelItem[]) => {
            // Filter out empty model names and convert back to object format
            const validModels = items.filter((model) => model.name.trim())
            const modelsConfig: Record<
                string,
                { type?: 'text' | 'image' | 'video'; description?: string }
            > = {}

            validModels.forEach((model) => {
                modelsConfig[model.name] = {
                    type: model.type,
                    description: model.description || ''
                }
            })

            onChange(modelsConfig)
        },
        [onChange]
    )

    const handleAddModel = () => {
        if (newModelName.trim()) {
            const newItems = [
                ...modelItems,
                {
                    name: newModelName.trim(),
                    type: 'image' as const,
                    description: newModelDescription.trim()
                },
            ]
            setModelItems(newItems)
            notifyChange(newItems)
            setNewModelName('')
            setNewModelDescription('')
            setOpenAddModelDialog(false)
        }
    }

    const handleEditModel = (model: ModelScopeModelItem) => {
        setEditingModel(model)
        setNewModelName(model.name)
        setNewModelDescription(model.description || '')
        setOpenAddModelDialog(true)
    }

    const handleUpdateModel = () => {
        if (editingModel && newModelName.trim()) {
            const newItems = modelItems.map(item =>
                item.name === editingModel.name
                    ? {
                        ...item,
                        name: newModelName.trim(),
                        description: newModelDescription.trim()
                    }
                    : item
            )
            setModelItems(newItems)
            notifyChange(newItems)
            setNewModelName('')
            setNewModelDescription('')
            setEditingModel(null)
            setOpenAddModelDialog(false)
        }
    }

    const handleRemoveModel = (index: number) => {
        const newItems = modelItems.filter((_, i) => i !== index)
        setModelItems(newItems)
        notifyChange(newItems)
    }

    const handleCloseDialog = () => {
        setOpenAddModelDialog(false)
        setEditingModel(null)
        setNewModelName('')
        setNewModelDescription('')
    }

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <Label>{label}</Label>
                <Dialog open={openAddModelDialog} onOpenChange={setOpenAddModelDialog}>
                    <DialogTrigger asChild>
                        <Button variant="secondary" size="sm">
                            <Plus className="h-4 w-4 mr-1" />
                            添加模型
                        </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-md">
                        <DialogHeader>
                            <DialogTitle>
                                {editingModel ? '编辑模型' : '添加ModelScope模型'}
                            </DialogTitle>
                        </DialogHeader>
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="model-name">模型名称</Label>
                                <Input
                                    id="model-name"
                                    type="text"
                                    placeholder="例如: MAILAND/majicflus_v1"
                                    value={newModelName}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault()
                                            editingModel ? handleUpdateModel() : handleAddModel()
                                        }
                                    }}
                                    onChange={(e) => setNewModelName(e.target.value)}
                                />
                                <p className="text-xs text-gray-500">
                                    输入魔搭社区的模型ID，支持图像生成模型
                                </p>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="model-description">模型描述</Label>
                                <Textarea
                                    id="model-description"
                                    placeholder="描述这个模型的用途和特点，让AI助手知道何时使用它..."
                                    value={newModelDescription}
                                    onChange={(e) => setNewModelDescription(e.target.value)}
                                    className="min-h-[100px] resize-none"
                                />
                                <p className="text-xs text-gray-500">
                                    详细描述帮助AI助手了解何时调用这个模型，比如：擅长的图像风格、适用场景等
                                </p>
                            </div>

                            <div className="flex gap-2">
                                <Button
                                    type="button"
                                    onClick={editingModel ? handleUpdateModel : handleAddModel}
                                    className="flex-1"
                                    disabled={!newModelName.trim()}
                                >
                                    {editingModel ? '更新模型' : '添加模型'}
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={handleCloseDialog}
                                >
                                    取消
                                </Button>
                            </div>
                        </div>
                    </DialogContent>
                </Dialog>
            </div>

            <div className="space-y-2">
                {modelItems.map((model, index) => (
                    <div key={index} className="border rounded-lg p-3 space-y-2">
                        <div className="flex items-start justify-between">
                            <div className="flex-1 space-y-1">
                                <div className="flex items-center gap-2">
                                    <p className="font-medium text-sm">{model.name}</p>
                                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                                        {model.type}
                                    </span>
                                </div>
                                {model.description && (
                                    <p className="text-sm text-gray-600 leading-relaxed">
                                        {model.description}
                                    </p>
                                )}
                            </div>
                            <div className="flex items-center gap-1 ml-2">
                                <Button
                                    type="button"
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => handleEditModel(model)}
                                    className="h-8 w-8"
                                >
                                    <PencilIcon className="h-4 w-4" />
                                </Button>
                                <Button
                                    type="button"
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => handleRemoveModel(index)}
                                    className="h-8 w-8 text-red-500 hover:text-red-700"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </Button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {modelItems.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                    <p className="text-sm">暂无配置的模型</p>
                    <p className="text-xs mt-1">点击上方"添加模型"按钮开始配置</p>
                </div>
            )}
        </div>
    )
}
