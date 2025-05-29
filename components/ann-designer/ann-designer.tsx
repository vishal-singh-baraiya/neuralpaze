"use client"

import type React from "react"

import { useCallback, useState, useEffect, useRef } from "react"
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
  BackgroundVariant,
  ReactFlowProvider,
} from "reactflow"
import "reactflow/dist/style.css"

import { ComponentLibrary } from "./component-library"
import { Toolbar } from "./toolbar"
import { ComponentEditor } from "./component-editor"
import { CodeModal } from "./code-modal"
import { CustomNode } from "./custom-node"
import { generatePyTorchCode } from "./utils/code-generator"
import type { ComponentType } from "./types"

const nodeTypes: NodeTypes = {
  customNode: CustomNode,
}

const initialNodes: Node[] = []
const initialEdges: Edge[] = []

// Complete ResizeObserver override and error suppression
const setupResizeObserverSuppression = () => {
  // Store original ResizeObserver
  const OriginalResizeObserver = window.ResizeObserver

  // Create a debounced ResizeObserver wrapper
  class SafeResizeObserver {
    private callback: ResizeObserverCallback
    private observer: ResizeObserver | null = null
    private timeoutId: NodeJS.Timeout | null = null

    constructor(callback: ResizeObserverCallback) {
      this.callback = callback
      try {
        this.observer = new OriginalResizeObserver((entries, observer) => {
          // Clear any existing timeout
          if (this.timeoutId) {
            clearTimeout(this.timeoutId)
          }

          // Debounce the callback to prevent loops
          this.timeoutId = setTimeout(() => {
            try {
              this.callback(entries, observer)
            } catch (error) {
              // Silently ignore ResizeObserver errors
              if (!error.message?.includes("ResizeObserver")) {
                console.warn("ResizeObserver callback error:", error)
              }
            }
          }, 16) // ~60fps
        })
      } catch (error) {
        console.warn("Failed to create ResizeObserver:", error)
      }
    }

    observe(target: Element, options?: ResizeObserverOptions) {
      try {
        this.observer?.observe(target, options)
      } catch (error) {
        // Silently ignore
      }
    }

    unobserve(target: Element) {
      try {
        this.observer?.unobserve(target)
      } catch (error) {
        // Silently ignore
      }
    }

    disconnect() {
      try {
        if (this.timeoutId) {
          clearTimeout(this.timeoutId)
          this.timeoutId = null
        }
        this.observer?.disconnect()
      } catch (error) {
        // Silently ignore
      }
    }
  }

  // Override global ResizeObserver
  window.ResizeObserver = SafeResizeObserver as any

  // Suppress all error types
  const originalError = console.error
  const originalWarn = console.warn

  console.error = (...args) => {
    const message = args[0]?.toString() || ""
    if (
      message.includes("ResizeObserver") ||
      message.includes("loop completed") ||
      message.includes("loop limit exceeded")
    ) {
      return
    }
    originalError.apply(console, args)
  }

  console.warn = (...args) => {
    const message = args[0]?.toString() || ""
    if (message.includes("ResizeObserver")) {
      return
    }
    originalWarn.apply(console, args)
  }

  // Handle window errors
  const handleError = (event: ErrorEvent) => {
    if (event.message?.includes("ResizeObserver") || event.error?.message?.includes("ResizeObserver")) {
      event.stopImmediatePropagation()
      event.preventDefault()
      return false
    }
  }

  // Handle unhandled promise rejections
  const handleRejection = (event: PromiseRejectionEvent) => {
    const reason = event.reason?.toString() || ""
    if (reason.includes("ResizeObserver")) {
      event.preventDefault()
      return false
    }
  }

  window.addEventListener("error", handleError, { capture: true, passive: false })
  window.addEventListener("unhandledrejection", handleRejection, { capture: true, passive: false })

  // Return cleanup function
  return () => {
    window.ResizeObserver = OriginalResizeObserver
    console.error = originalError
    console.warn = originalWarn
    window.removeEventListener("error", handleError, true)
    window.removeEventListener("unhandledrejection", handleRejection, true)
  }
}

function ANNDesignerFlow() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [showCodeModal, setShowCodeModal] = useState(false)
  const [generatedCode, setGeneratedCode] = useState("")
  const [isReady, setIsReady] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

  // Setup error suppression and delayed initialization
  useEffect(() => {
    // Setup ResizeObserver suppression immediately
    cleanupRef.current = setupResizeObserverSuppression()

    // Delay React Flow initialization to prevent conflicts
    const initTimer = setTimeout(() => {
      setIsReady(true)
    }, 200)

    return () => {
      clearTimeout(initTimer)
      if (cleanupRef.current) {
        cleanupRef.current()
      }
    }
  }, [])

  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            animated: true,
            style: {
              stroke: "#60A5FA",
              strokeWidth: 3,
              filter: "drop-shadow(0 0 6px rgba(96, 165, 250, 0.5))",
            },
            markerEnd: {
              type: "arrowclosed",
              color: "#60A5FA",
            },
          },
          eds,
        ),
      ),
    [setEdges],
  )

  const addComponent = useCallback(
    (componentType: ComponentType) => {
      const newNode: Node = {
        id: `${Date.now()}-${Math.random()}`,
        type: "customNode",
        position: { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 },
        data: {
          type: componentType.type,
          icon: componentType.icon,
          color: componentType.color,
          params: { ...componentType.params },
          description: componentType.description,
        },
        dragHandle: ".drag-handle",
      }
      setNodes((nds) => nds.concat(newNode))
    },
    [setNodes],
  )

  const updateNode = useCallback(
    (id: string, updates: any) => {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === id) {
            return { ...node, data: { ...node.data, ...updates } }
          }
          return node
        }),
      )
      if (selectedNode?.id === id) {
        setSelectedNode((prev) => (prev ? { ...prev, data: { ...prev.data, ...updates } } : null))
      }
    },
    [setNodes, selectedNode],
  )

  const deleteNode = useCallback(
    (id: string) => {
      setNodes((nds) => nds.filter((node) => node.id !== id))
      setEdges((eds) => eds.filter((edge) => edge.source !== id && edge.target !== id))
      if (selectedNode?.id === id) {
        setSelectedNode(null)
      }
    },
    [setNodes, setEdges, selectedNode],
  )

  const duplicateNode = useCallback(
    (id: string) => {
      const node = nodes.find((n) => n.id === id)
      if (node) {
        const newNode: Node = {
          ...node,
          id: `${Date.now()}-${Math.random()}`,
          position: { x: node.position.x + 50, y: node.position.y + 50 },
        }
        setNodes((nds) => nds.concat(newNode))
      }
    },
    [nodes, setNodes],
  )

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedNode(null)
  }, [])

  const exportNetwork = () => {
    const networkData = {
      nodes,
      edges,
      metadata: {
        created: new Date().toISOString(),
        version: "2.0",
      },
    }
    const blob = new Blob([JSON.stringify(networkData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "neural_network.json"
    a.click()
    URL.revokeObjectURL(url)
  }

  const importNetwork = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const networkData = JSON.parse(e.target?.result as string)
          setNodes(networkData.nodes || [])
          setEdges(networkData.edges || [])
        } catch (error) {
          alert("Error importing network: Invalid file format")
        }
      }
      reader.readAsText(file)
    }
    event.target.value = ""
  }

  const generateCode = () => {
    try {
      const components = nodes.map((node) => ({
        id: node.id,
        type: node.data.type,
        x: node.position.x,
        y: node.position.y,
        params: node.data.params,
        color: node.data.color,
        icon: node.data.icon,
        inputs: [],
        outputs: [],
      }))

      const connections = edges.map((edge) => ({
        id: edge.id,
        from: edge.source,
        to: edge.target,
        fromPort: "output",
        toPort: "input",
      }))

      const code = generatePyTorchCode(components, connections)
      setGeneratedCode(code)
      setShowCodeModal(true)
    } catch (error) {
      console.error("Error generating code:", error)
      alert(`Error generating code: ${error}`)
    }
  }

  // Show loading screen while initializing
  if (!isReady) {
    return (
      <div className="w-full h-screen bg-black flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <div className="text-white text-lg">Initializing NeuralPaze...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-screen bg-black flex overflow-hidden">
      {/* Component Library Sidebar */}
      <ComponentLibrary onAddComponent={addComponent} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <Toolbar onExport={exportNetwork} onImport={importNetwork} onGenerateCode={generateCode} />

        {/* React Flow Canvas */}
        <div ref={containerRef} className="flex-1 bg-black">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            className="bg-black"
            style={{ background: "black" }}
            defaultViewport={{ x: 0, y: 0, zoom: 1 }}
            minZoom={0.2}
            maxZoom={2}
            fitView={false}
            connectionLineStyle={{
              stroke: "#60A5FA",
              strokeWidth: 3,
            }}
            defaultEdgeOptions={{
              animated: true,
              style: {
                stroke: "#60A5FA",
                strokeWidth: 3,
              },
            }}
            proOptions={{ hideAttribution: true }}
            onError={(id, message) => {
              // Suppress React Flow errors
              if (!message.includes("ResizeObserver")) {
                console.warn("React Flow error:", id, message)
              }
            }}
          >
            <Controls
              className="!bg-black/80 !backdrop-blur-xl !border !border-white/20 !rounded-lg"
              style={{
                backgroundColor: "rgba(0,0,0,0.8)",
              }}
            />
            <MiniMap
              className="!bg-black/80 !backdrop-blur-xl !border !border-white/20 !rounded-lg"
              style={{
                backgroundColor: "rgba(0,0,0,0.8)",
              }}
              nodeColor="#60A5FA"
              maskColor="rgba(0,0,0,0.8)"
            />
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="rgba(255,255,255,0.1)"
              style={{ backgroundColor: "black" }}
            />
          </ReactFlow>
        </div>
      </div>

      {/* Component Editor Panel */}
      {selectedNode && (
        <ComponentEditor
          node={selectedNode}
          onUpdate={updateNode}
          onClose={() => setSelectedNode(null)}
          onDuplicate={duplicateNode}
          onDelete={deleteNode}
        />
      )}

      {/* Code Generation Modal */}
      {showCodeModal && <CodeModal code={generatedCode} onClose={() => setShowCodeModal(false)} />}
    </div>
  )
}

export function ANNDesigner() {
  return (
    <ReactFlowProvider>
      <ANNDesignerFlow />
    </ReactFlowProvider>
  )
}
