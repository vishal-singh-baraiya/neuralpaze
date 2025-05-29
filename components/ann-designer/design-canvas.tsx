"use client"
import { forwardRef } from "react"
import type { Component, Connection } from "./types"
import { ComponentNode } from "./component-node"
import { ConnectionLine } from "./connection-line"

interface DesignCanvasProps {
  components: Component[]
  connections: Connection[]
  selectedComponent: Component | null
  showGrid: boolean
  scale: number
  onComponentClick: (component: Component) => void
  onStartConnection: (componentId: string | number, port: string, type: string) => void
  onUpdateComponent: (id: string | number, updates: Partial<Component>) => void
}

export const DesignCanvas = forwardRef<HTMLDivElement, DesignCanvasProps>(
  (
    {
      components,
      connections,
      selectedComponent,
      showGrid,
      scale,
      onComponentClick,
      onStartConnection,
      onUpdateComponent,
    },
    ref,
  ) => {
    const handleUpdatePosition = (id: string | number, x: number, y: number) => {
      onUpdateComponent(id, { x, y })
    }

    return (
      <div
        ref={ref}
        className="flex-1 relative overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900"
        style={{
          backgroundImage: showGrid ? `radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px)` : "none",
          backgroundSize: showGrid ? `${20 * scale}px ${20 * scale}px` : "auto",
        }}
      >
        <div
          className="absolute inset-0 w-full h-full"
          style={{
            transform: `scale(${scale})`,
            transformOrigin: "top left",
            width: `${100 / scale}%`,
            height: `${100 / scale}%`,
          }}
        >
          {/* Connections */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
            {connections.map((conn) => {
              const fromComp = components.find((c) => c.id === conn.from)
              const toComp = components.find((c) => c.id === conn.to)
              if (!fromComp || !toComp) return null

              return <ConnectionLine key={conn.id} from={fromComp} to={toComp} />
            })}
            <defs>
              <marker id="arrowhead" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto">
                <polygon points="0 0, 12 4, 0 8" fill="rgba(96, 165, 250, 0.8)" />
              </marker>
            </defs>
          </svg>

          {/* Components */}
          <div className="relative w-full h-full" style={{ zIndex: 2 }}>
            {components.map((component) => (
              <ComponentNode
                key={component.id}
                component={component}
                isSelected={selectedComponent?.id === component.id}
                onClick={onComponentClick}
                onStartConnection={onStartConnection}
                onUpdatePosition={handleUpdatePosition}
                scale={scale}
              />
            ))}
          </div>
        </div>
      </div>
    )
  },
)

DesignCanvas.displayName = "DesignCanvas"
