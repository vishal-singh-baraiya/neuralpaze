"use client"
import { ScrollArea } from "@/components/ui/scroll-area"
import { componentLibrary } from "./data/component-library"
import type { ComponentType } from "./types"

interface ComponentLibraryProps {
  onAddComponent: (componentType: ComponentType) => void
}

export function ComponentLibrary({ onAddComponent }: ComponentLibraryProps) {
  return (
    <div className="w-80 bg-black/40 backdrop-blur-xl border-r border-white/10 flex flex-col">
      <div className="p-6 border-b border-white/10">
        <h2 className="font-semibold text-xl text-white">Component Library</h2>
        <p className="text-sm text-gray-400 mt-1">Drag components to canvas</p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          {Object.entries(componentLibrary).map(([category, items]) => (
            <div key={category} className="mb-6">
              <h3 className="font-medium text-sm text-gray-300 mb-3 uppercase tracking-wider">{category}</h3>
              <div className="space-y-2">
                {items.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => onAddComponent(item)}
                    className="w-full text-left p-3 rounded-lg bg-white/5 hover:bg-white/10 
                             border border-white/10 hover:border-white/20 transition-all duration-200
                             backdrop-blur-sm group"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-8 h-8 rounded-md flex items-center justify-center text-lg font-mono
                                 bg-gradient-to-br from-white/10 to-white/5"
                        style={{ color: item.color }}
                      >
                        {item.icon}
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-medium text-white group-hover:text-gray-100">{item.type}</div>
                        {item.description && <div className="text-xs text-gray-400 mt-1">{item.description}</div>}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
