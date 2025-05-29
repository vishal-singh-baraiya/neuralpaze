"use client"

import { useState, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { X, Search, Layers, Zap, Brain, Sparkles } from "lucide-react"
import { modelTemplates } from "../data/model-templates"
import type { ModelTemplate } from "../types"

interface ModelTemplatesProps {
  onLoadTemplate: (template: ModelTemplate) => void
  onClose: () => void
}

const categoryIcons = {
  "Basic Networks": Layers,
  Transformers: Brain,
  "Advanced Architectures": Zap,
  "Generative Models": Sparkles,
}

const categoryColors = {
  "Basic Networks": "from-blue-500/20 to-blue-600/20 border-blue-500/30",
  Transformers: "from-purple-500/20 to-purple-600/20 border-purple-500/30",
  "Advanced Architectures": "from-orange-500/20 to-orange-600/20 border-orange-500/30",
  "Generative Models": "from-pink-500/20 to-pink-600/20 border-pink-500/30",
}

export function ModelTemplates({ onLoadTemplate, onClose }: ModelTemplatesProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>("all")
  const [searchQuery, setSearchQuery] = useState("")

  const categories = useMemo(() => {
    return ["all", ...Object.keys(modelTemplates)]
  }, [])

  const filteredTemplates = useMemo(() => {
    let templates =
      selectedCategory === "all" ? Object.values(modelTemplates).flat() : modelTemplates[selectedCategory] || []

    if (searchQuery.trim()) {
      templates = templates.filter(
        (template) =>
          template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          template.description.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    }

    return templates
  }, [selectedCategory, searchQuery])

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-900/95 to-black/95 backdrop-blur-xl border border-white/20 rounded-2xl w-full max-w-6xl h-[85vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 flex items-center justify-center">
              <Layers className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Model Templates</h2>
              <p className="text-sm text-gray-400">Choose from pre-built neural network architectures</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="text-gray-400 hover:text-white hover:bg-white/10 rounded-xl"
          >
            <X size={20} />
          </Button>
        </div>

        {/* Search and Filters */}
        <div className="p-6 border-b border-white/10 space-y-4">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-white/5 border-white/20 text-white placeholder:text-gray-500 rounded-xl"
            />
          </div>

          {/* Category Filters */}
          <div className="flex flex-wrap gap-2">
            {categories.map((category) => {
              const isActive = selectedCategory === category
              const Icon = category !== "all" ? categoryIcons[category as keyof typeof categoryIcons] : Layers

              return (
                <Button
                  key={category}
                  variant={isActive ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedCategory(category)}
                  className={`rounded-xl transition-all duration-200 ${
                    isActive
                      ? "bg-white/15 text-white border border-white/20 shadow-lg"
                      : "text-gray-400 hover:text-white hover:bg-white/10"
                  }`}
                >
                  {Icon && <Icon className="w-4 h-4 mr-2" />}
                  {category === "all" ? "All Templates" : category}
                </Button>
              )
            })}
          </div>
        </div>

        {/* Templates Grid */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full overflow-y-auto p-6">
            {filteredTemplates.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-16 h-16 rounded-full bg-gray-800/50 flex items-center justify-center mb-4">
                  <Search className="w-8 h-8 text-gray-500" />
                </div>
                <h3 className="text-lg font-medium text-gray-400 mb-2">No templates found</h3>
                <p className="text-sm text-gray-500">Try adjusting your search or category filter</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredTemplates.map((template) => {
                  const Icon = categoryIcons[template.category as keyof typeof categoryIcons] || Layers
                  const colorClass =
                    categoryColors[template.category as keyof typeof categoryColors] || categoryColors["Basic Networks"]

                  return (
                    <div
                      key={template.id}
                      className={`group relative bg-gradient-to-br ${colorClass} backdrop-blur-sm rounded-2xl p-6 cursor-pointer transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:shadow-white/10`}
                      onClick={() => onLoadTemplate(template)}
                    >
                      {/* Template Header */}
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-white/10 backdrop-blur-sm flex items-center justify-center">
                            <Icon className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-white group-hover:text-gray-100 transition-colors">
                              {template.name}
                            </h3>
                            <Badge variant="outline" className="bg-white/10 text-white/80 border-white/20 text-xs mt-1">
                              {template.category}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      {/* Template Description */}
                      <p className="text-sm text-white/80 mb-4 leading-relaxed">{template.description}</p>

                      {/* Template Stats */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                            <span className="text-xs text-white/70">{template.components.length} layers</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-green-400"></div>
                            <span className="text-xs text-white/70">{template.connections.length} connections</span>
                          </div>
                        </div>
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-white/80 hover:text-white hover:bg-white/10 rounded-lg"
                          >
                            Load Template
                          </Button>
                        </div>
                      </div>

                      {/* Hover Effect Overlay */}
                      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/10 bg-black/20">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-400">
              {filteredTemplates.length} template{filteredTemplates.length !== 1 ? "s" : ""} available
            </p>
            <Button
              variant="ghost"
              onClick={onClose}
              className="text-gray-400 hover:text-white hover:bg-white/10 rounded-xl"
            >
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
