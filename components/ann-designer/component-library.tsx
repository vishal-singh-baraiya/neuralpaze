"use client"
import { useState, useMemo } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Search, X } from "lucide-react"
import { componentLibrary } from "./data/component-library"
import type { ComponentType } from "./types"

interface ComponentLibraryProps {
  onAddComponent: (componentType: ComponentType) => void
}

export function ComponentLibrary({ onAddComponent }: ComponentLibraryProps) {
  const [searchQuery, setSearchQuery] = useState("")

  // Filter components based on search query
  const filteredLibrary = useMemo(() => {
    if (!searchQuery.trim()) {
      return componentLibrary
    }

    const query = searchQuery.toLowerCase().trim()
    const result: Record<string, ComponentType[]> = {}

    Object.entries(componentLibrary).forEach(([category, items]) => {
      const filteredItems = items.filter(
        (item) =>
          item.type.toLowerCase().includes(query) ||
          (item.description && item.description.toLowerCase().includes(query)),
      )

      if (filteredItems.length > 0) {
        result[category] = filteredItems
      }
    })

    return result
  }, [searchQuery])

  // Check if any components match the search
  const hasResults = useMemo(() => {
    return Object.values(filteredLibrary).some((items) => items.length > 0)
  }, [filteredLibrary])

  return (
    <div className="w-80 bg-black/40 backdrop-blur-xl border-r border-white/10 flex flex-col">
      <div className="p-6 border-b border-white/10">
        <h2 className="font-semibold text-xl text-white">Component Library</h2>
        <p className="text-sm text-gray-400 mt-1">Drag components to canvas</p>
      </div>

      {/* Search Section */}
      <div className="p-4 border-b border-white/10">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search components..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-white/5 border-white/20 text-white placeholder:text-gray-500 rounded-lg"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-2 top-1/2 transform -translate-y-1/2 h-6 w-6 text-gray-400 hover:text-white"
              onClick={() => setSearchQuery("")}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <ScrollArea className="flex-1">
        {!hasResults && searchQuery ? (
          <div className="p-8 text-center">
            <div className="w-12 h-12 rounded-full bg-gray-800/50 flex items-center justify-center mx-auto mb-3">
              <Search className="w-6 h-6 text-gray-500" />
            </div>
            <h3 className="text-gray-400 font-medium mb-1">No components found</h3>
            <p className="text-sm text-gray-500">Try a different search term</p>
          </div>
        ) : (
          <div className="p-4">
            {Object.entries(filteredLibrary).map(([category, items]) => (
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
        )}
      </ScrollArea>
    </div>
  )
}
