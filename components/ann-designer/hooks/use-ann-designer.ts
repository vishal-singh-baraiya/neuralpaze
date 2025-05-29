"use client"

import type React from "react"
import { useState, useRef, useCallback } from "react"
import type { Component, Connection } from "../types"
import { generatePyTorchCode } from "../utils/code-generator"

export function useANNDesigner() {
  const [components, setComponents] = useState<Component[]>([])
  const [connections, setConnections] = useState<Connection[]>([])
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null)
  const [showGrid, setShowGrid] = useState(true)
  const [scale, setScale] = useState(1)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionStart, setConnectionStart] = useState<any>(null)
  const [showCodeModal, setShowCodeModal] = useState(false)
  const [generatedCode, setGeneratedCode] = useState("")
  const canvasRef = useRef<HTMLDivElement>(null)

  const addComponent = useCallback((componentType: any, x?: number, y?: number) => {
    const newComponent: Component = {
      id: Date.now() + Math.random(),
      type: componentType.type,
      x: x ?? Math.random() * 400 + 100,
      y: y ?? Math.random() * 400 + 100,
      params: { ...componentType.params },
      color: componentType.color,
      icon: componentType.icon,
      inputs: [],
      outputs: [],
    }
    setComponents((prev) => [...prev, newComponent])
  }, [])

  const updateComponent = useCallback((id: string | number, updates: Partial<Component>) => {
    setComponents((prev) => prev.map((comp) => (comp.id === id ? { ...comp, ...updates } : comp)))
    setSelectedComponent((prev) => (prev?.id === id ? ({ ...prev, ...updates } as Component) : prev))
  }, [])

  const deleteComponent = useCallback(
    (id: string | number) => {
      setComponents((prev) => prev.filter((comp) => comp.id !== id))
      setConnections((prev) => prev.filter((conn) => conn.from !== id && conn.to !== id))
      if (selectedComponent?.id === id) {
        setSelectedComponent(null)
      }
    },
    [selectedComponent],
  )

  const duplicateComponent = useCallback(
    (id: string | number) => {
      const component = components.find((c) => c.id === id)
      if (component) {
        addComponent(
          {
            type: component.type,
            icon: component.icon,
            color: component.color,
            params: component.params,
          },
          component.x + 50,
          component.y + 50,
        )
      }
    },
    [components, addComponent],
  )

  const startConnection = (componentId: string | number, port: string, type: string) => {
    if (isConnecting) {
      if (connectionStart && connectionStart.id !== componentId) {
        const newConnection: Connection = {
          id: Date.now(),
          from: connectionStart.type === "output" ? connectionStart.id : componentId,
          to: connectionStart.type === "input" ? connectionStart.id : componentId,
          fromPort: connectionStart.type === "output" ? connectionStart.port : port,
          toPort: connectionStart.type === "input" ? connectionStart.port : port,
        }
        setConnections((prev) => [...prev, newConnection])
      }
      setIsConnecting(false)
      setConnectionStart(null)
    } else {
      setIsConnecting(true)
      setConnectionStart({ id: componentId, port, type })
    }
  }

  const exportNetwork = () => {
    const networkData = {
      components,
      connections,
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
          setComponents(networkData.components || [])
          setConnections(networkData.connections || [])
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
      const code = generatePyTorchCode(components, connections)
      setGeneratedCode(code)
      setShowCodeModal(true)
    } catch (error) {
      console.error("Error generating code:", error)
      alert(`Error generating code: ${error}`)
    }
  }

  return {
    components,
    connections,
    selectedComponent,
    showGrid,
    scale,
    isConnecting,
    connectionStart,
    showCodeModal,
    generatedCode,
    canvasRef,
    addComponent,
    updateComponent,
    deleteComponent,
    duplicateComponent,
    setSelectedComponent,
    setShowGrid,
    setScale,
    startConnection,
    exportNetwork,
    importNetwork,
    generateCode,
    setShowCodeModal,
  }
}
