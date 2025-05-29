"use client"

import { Handle, Position, type NodeProps } from "reactflow"
import { memo } from "react"

interface CustomNodeData {
  type: string
  icon: string
  color: string
  params: Record<string, any>
  description?: string
}

export const CustomNode = memo(({ data, selected }: NodeProps<CustomNodeData>) => {
  return (
    <div
      className={`bg-black/90 backdrop-blur-xl border-2 rounded-xl p-4 min-w-[220px] transition-all duration-200 ${
        selected ? "border-blue-400 shadow-lg shadow-blue-400/20" : "border-white/20"
      } hover:border-white/40 cursor-pointer`}
      style={{
        borderColor: selected ? "#60A5FA" : "rgba(255,255,255,0.2)",
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="w-4 h-4 !bg-gradient-to-r !from-green-400 !to-green-500 !border-0 hover:!scale-125 !transition-transform !shadow-lg !shadow-green-400/30"
        style={{
          background: "linear-gradient(to right, #34d399, #10b981)",
          border: "none",
        }}
      />

      <div className="drag-handle flex items-center gap-3 mb-3 cursor-grab active:cursor-grabbing">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center text-xl font-mono
                     bg-gradient-to-br from-white/10 to-white/5 flex-shrink-0 shadow-lg"
          style={{ color: data.color }}
        >
          {data.icon}
        </div>
        <div className="flex-1">
          <div className="font-medium text-white text-sm">{data.type}</div>
          {data.description && <div className="text-xs text-gray-400 mt-1">{data.description}</div>}
        </div>
      </div>

      {/* Parameters preview */}
      <div className="text-xs text-gray-400 space-y-1 bg-white/5 rounded-lg p-3">
        {Object.entries(data.params)
          .slice(0, 3)
          .map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="opacity-70 truncate font-medium">{key}:</span>
              <span className="text-gray-300 truncate max-w-[80px] font-mono">
                {typeof value === "string" ? value.slice(0, 8) : String(value)}
              </span>
            </div>
          ))}
        {Object.keys(data.params).length > 3 && (
          <div className="text-gray-500 text-center font-medium">+{Object.keys(data.params).length - 3} more</div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="w-4 h-4 !bg-gradient-to-r !from-red-400 !to-red-500 !border-0 hover:!scale-125 !transition-transform !shadow-lg !shadow-red-400/30"
        style={{
          background: "linear-gradient(to right, #f87171, #ef4444)",
          border: "none",
        }}
      />
    </div>
  )
})

CustomNode.displayName = "CustomNode"
