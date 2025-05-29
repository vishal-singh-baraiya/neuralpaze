"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { X, Info, Layers, Code, Settings } from "lucide-react"

interface ModelInfoProps {
  onClose: () => void
  nodeCount: number
  edgeCount: number
}

export function ModelInfo({ onClose, nodeCount, edgeCount }: ModelInfoProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "settings" | "code">("overview")

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-black/90 border border-white/10 rounded-lg w-full max-w-3xl max-h-[80vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/30 flex items-center justify-center">
              <Info className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Model Information</h2>
              <p className="text-sm text-gray-400">Details about your neural network</p>
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

        {/* Tabs */}
        <div className="flex border-b border-white/10">
          <Button
            variant="ghost"
            className={`rounded-none px-6 py-3 ${
              activeTab === "overview" ? "text-white border-b-2 border-blue-500" : "text-gray-400"
            }`}
            onClick={() => setActiveTab("overview")}
          >
            <Layers className="w-4 h-4 mr-2" />
            Overview
          </Button>
          <Button
            variant="ghost"
            className={`rounded-none px-6 py-3 ${
              activeTab === "settings" ? "text-white border-b-2 border-blue-500" : "text-gray-400"
            }`}
            onClick={() => setActiveTab("settings")}
          >
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
          <Button
            variant="ghost"
            className={`rounded-none px-6 py-3 ${
              activeTab === "code" ? "text-white border-b-2 border-blue-500" : "text-gray-400"
            }`}
            onClick={() => setActiveTab("code")}
          >
            <Code className="w-4 h-4 mr-2" />
            Code Preview
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === "overview" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <div className="text-sm text-gray-400 mb-1">Components</div>
                  <div className="text-2xl font-bold text-white">{nodeCount}</div>
                </div>
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <div className="text-sm text-gray-400 mb-1">Connections</div>
                  <div className="text-2xl font-bold text-white">{edgeCount}</div>
                </div>
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-medium text-white mb-3">Model Structure</h3>
                <div className="text-sm text-gray-400">
                  {nodeCount === 0 ? (
                    <p>No components added yet. Start by adding components from the library.</p>
                  ) : (
                    <p>
                      Your model contains {nodeCount} components connected with {edgeCount} connections. Use the Code
                      Preview tab to see the generated PyTorch code.
                    </p>
                  )}
                </div>
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-medium text-white mb-3">Tips</h3>
                <ul className="text-sm text-gray-400 space-y-2 list-disc pl-5">
                  <li>Drag components from the library to the canvas</li>
                  <li>Connect components by clicking on their input/output ports</li>
                  <li>Click on a component to edit its parameters</li>
                  <li>Use the toolbar to export your model or generate code</li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === "settings" && (
            <div className="space-y-6">
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-medium text-white mb-3">Display Settings</h3>
                <p className="text-sm text-gray-400 mb-4">Configure how the model is displayed on the canvas.</p>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Show grid</span>
                    <Button variant="outline" size="sm" className="h-8 px-3">
                      Enabled
                    </Button>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Show component details</span>
                    <Button variant="outline" size="sm" className="h-8 px-3">
                      Enabled
                    </Button>
                  </div>
                </div>
              </div>

              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-medium text-white mb-3">Code Generation</h3>
                <p className="text-sm text-gray-400 mb-4">Configure code generation settings.</p>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Framework</span>
                    <Button variant="outline" size="sm" className="h-8 px-3">
                      PyTorch
                    </Button>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Include comments</span>
                    <Button variant="outline" size="sm" className="h-8 px-3">
                      Enabled
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "code" && (
            <div className="bg-white/5 rounded-xl p-4 border border-white/10">
              <h3 className="text-lg font-medium text-white mb-3">Code Preview</h3>
              <p className="text-sm text-gray-400 mb-4">
                Preview of the generated PyTorch code. Use the "Generate Code" button in the toolbar for the full code.
              </p>
              <div className="bg-gray-900/50 rounded-lg border border-white/10 p-4 font-mono text-sm text-gray-300 overflow-x-auto">
                {nodeCount === 0 ? (
                  <span className="text-gray-500">No components added yet. Add components to generate code.</span>
                ) : (
                  <pre>{`import torch
import torch.nn as nn

class GeneratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # ${nodeCount} components will be initialized here
        # ...

    def forward(self, x):
        # Forward pass through ${nodeCount} components with ${edgeCount} connections
        # ...
        return x`}</pre>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/10 bg-black/20">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-400">
              {nodeCount} component{nodeCount !== 1 ? "s" : ""}, {edgeCount} connection
              {edgeCount !== 1 ? "s" : ""}
            </p>
            <Button onClick={onClose} className="bg-blue-600 hover:bg-blue-700 text-white rounded-xl">
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
