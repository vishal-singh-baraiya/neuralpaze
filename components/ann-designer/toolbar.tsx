"use client"

import type React from "react"
import { Button } from "@/components/ui/button"
import { Download, Upload, Play } from "lucide-react"

interface ToolbarProps {
  onExport: () => void
  onImport: (event: React.ChangeEvent<HTMLInputElement>) => void
  onGenerateCode: () => void
}

export function Toolbar({ onExport, onImport, onGenerateCode }: ToolbarProps) {
  return (
    <div className="bg-black/60 backdrop-blur-xl border-b border-white/10 p-4 flex items-center gap-6">
      <div className="flex items-center gap-2">
        <Button
          onClick={onExport}
          variant="ghost"
          size="sm"
          className="text-white hover:bg-green-500/20 hover:text-green-400"
        >
          <Download size={16} />
          <span className="ml-2">Export</span>
        </Button>

        <label>
          <Button
            variant="ghost"
            size="sm"
            className="text-white hover:bg-blue-500/20 hover:text-blue-400 cursor-pointer"
            asChild
          >
            <span>
              <Upload size={16} />
              <span className="ml-2">Import</span>
            </span>
          </Button>
          <input type="file" accept=".json" onChange={onImport} className="hidden" />
        </label>

        <Button
          onClick={onGenerateCode}
          variant="ghost"
          size="sm"
          className="text-white hover:bg-purple-500/20 hover:text-purple-400"
        >
          <Play size={16} />
          <span className="ml-2">Generate Code</span>
        </Button>
      </div>
    </div>
  )
}
