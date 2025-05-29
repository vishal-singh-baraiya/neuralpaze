"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Switch } from "@/components/ui/switch"
import { Copy, Trash2, X } from "lucide-react"
import type { Node } from "reactflow"
import { useState, useEffect } from "react"

interface ComponentEditorProps {
  node: Node
  onUpdate: (id: string, updates: any) => void
  onClose: () => void
  onDuplicate: (id: string) => void
  onDelete: (id: string) => void
}

export function ComponentEditor({ node, onUpdate, onClose, onDuplicate, onDelete }: ComponentEditorProps) {
  const [localParams, setLocalParams] = useState(node.data.params)

  useEffect(() => {
    setLocalParams(node.data.params)
  }, [node.data.params])

  const updateParam = (key: string, value: any) => {
    const newParams = { ...localParams, [key]: value }
    setLocalParams(newParams)
    onUpdate(node.id, { params: newParams })
  }

  const handleInputChange = (key: string, e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    updateParam(key, e.target.value)
  }

  const handleNumberChange = (key: string, e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value === "" ? 0 : Number.parseFloat(e.target.value) || 0
    updateParam(key, value)
  }

  const handleSwitchChange = (key: string, checked: boolean) => {
    updateParam(key, checked)
  }

  return (
    <div className="w-96 bg-black/80 backdrop-blur-xl border-l border-white/10 flex flex-col max-h-screen">
      <div className="p-6 border-b border-white/10 flex-shrink-0">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center text-lg font-mono
                         bg-gradient-to-br from-white/10 to-white/5"
              style={{ color: node.data.color }}
            >
              {node.data.icon}
            </div>
            <div>
              <h3 className="font-semibold text-lg text-white">{node.data.type}</h3>
              {node.data.description && <p className="text-sm text-gray-400">{node.data.description}</p>}
            </div>
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="text-gray-400 hover:text-white hover:bg-white/10"
          >
            <X size={16} />
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-4">
          {node.data.type === "CustomLayer" ? (
            <>
              <div>
                <Label className="text-gray-300 mb-2 block">Class Name</Label>
                <Input
                  value={localParams.name || ""}
                  onChange={(e) => handleInputChange("name", e)}
                  className="bg-white/5 border-white/20 text-white placeholder:text-gray-500"
                  placeholder="e.g., MyCustomLayer"
                />
              </div>
              <div>
                <Label className="text-gray-300 mb-2 block">PyTorch Code</Label>
                <Textarea
                  value={localParams.code || ""}
                  onChange={(e) => handleInputChange("code", e)}
                  className="bg-white/5 border-white/20 text-white placeholder:text-gray-500 font-mono text-sm min-h-[200px]"
                  placeholder={`# Example:
self.conv1 = nn.Conv2d(3, 64, 3)
self.conv2 = nn.Conv2d(64, 128, 3)
self.fc = nn.Linear(128, 10)

def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    return self.fc(x)`}
                />
              </div>
            </>
          ) : (
            Object.entries(localParams).map(([key, value]) => (
              <div key={key}>
                <Label className="text-gray-300 mb-2 block">
                  {key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                </Label>
                {typeof value === "boolean" ? (
                  <div className="flex items-center space-x-2">
                    <Switch checked={value} onCheckedChange={(checked) => handleSwitchChange(key, checked)} />
                    <span className="text-sm text-gray-400">{value ? "True" : "False"}</span>
                  </div>
                ) : key === "code" ? (
                  <Textarea
                    value={value || ""}
                    onChange={(e) => handleInputChange(key, e)}
                    className="bg-white/5 border-white/20 text-white placeholder:text-gray-500 font-mono text-sm"
                    rows={4}
                  />
                ) : typeof value === "number" ? (
                  <Input
                    type="number"
                    value={value || ""}
                    onChange={(e) => handleNumberChange(key, e)}
                    className="bg-white/5 border-white/20 text-white placeholder:text-gray-500"
                    step="any"
                  />
                ) : (
                  <Input
                    type="text"
                    value={value || ""}
                    onChange={(e) => handleInputChange(key, e)}
                    className="bg-white/5 border-white/20 text-white placeholder:text-gray-500"
                  />
                )}
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      <div className="p-6 border-t border-white/10 flex gap-3 flex-shrink-0">
        <Button
          onClick={() => onDuplicate(node.id)}
          variant="ghost"
          size="sm"
          className="flex-1 text-blue-400 hover:text-blue-300 hover:bg-blue-500/20"
        >
          <Copy size={16} />
          <span className="ml-2">Duplicate</span>
        </Button>
        <Button
          onClick={() => onDelete(node.id)}
          variant="ghost"
          size="sm"
          className="flex-1 text-red-400 hover:text-red-300 hover:bg-red-500/20"
        >
          <Trash2 size={16} />
          <span className="ml-2">Delete</span>
        </Button>
      </div>
    </div>
  )
}
