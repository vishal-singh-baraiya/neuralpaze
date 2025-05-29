"use client"

import type React from "react"

import { useCallback, useState } from "react"
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

export function ANNDesigner() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [showCodeModal, setShowCodeModal] = useState(false)
  const [generatedCode, setGeneratedCode] = useState("")

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

  return (
    <div className="w-full h-screen bg-black flex overflow-hidden">
      {/* Component Library Sidebar */}
      <ComponentLibrary onAddComponent={addComponent} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <Toolbar onExport={exportNetwork} onImport={importNetwork} onGenerateCode={generateCode} />

        {/* React Flow Canvas */}
        <div className="flex-1 bg-black">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            className="bg-black"
            style={{ background: "black" }}
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
