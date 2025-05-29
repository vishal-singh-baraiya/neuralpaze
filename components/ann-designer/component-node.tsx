"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import type { Component } from "./types"

interface ComponentNodeProps {
  component: Component
  isSelected: boolean
  onClick: (component: Component) => void
  onStartConnection: (componentId: string | number, port: string, type: string) => void
  onUpdatePosition: (id: string | number, x: number, y: number) => void
  scale: number
}

export function ComponentNode({
  component,
  isSelected,
  onClick,
  onStartConnection,
  onUpdatePosition,
  scale,
}: ComponentNodeProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, compX: 0, compY: 0 })
  const nodeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return

      e.preventDefault()

      // Calculate the movement delta
      const deltaX = e.clientX - dragStart.x
      const deltaY = e.clientY - dragStart.y

      // Apply scale factor and calculate new position
      const newX = Math.max(0, dragStart.compX + deltaX / scale)
      const newY = Math.max(0, dragStart.compY + deltaY / scale)

      // Update position immediately
      onUpdatePosition(component.id, newX, newY)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }

    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove, { passive: false })
      document.addEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = "grabbing"
      document.body.style.userSelect = "none"
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isDragging, dragStart, component.id, onUpdatePosition, scale])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return // Only left click

    e.preventDefault()
    e.stopPropagation()

    setIsDragging(true)
    setDragStart({
      x: e.clientX,
      y: e.clientY,
      compX: component.x,
      compY: component.y,
    })
    onClick(component)
  }

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onClick(component)
  }

  return (
    <div
      ref={nodeRef}
      className={`absolute bg-black/80 backdrop-blur-xl border-2 rounded-xl p-4 select-none
                  transition-all duration-200 hover:bg-black/90 min-w-[220px] ${
                    isSelected ? "border-blue-400 shadow-lg shadow-blue-400/20" : "border-white/20"
                  } ${isDragging ? "cursor-grabbing z-50 scale-105" : "cursor-grab"}`}
      style={{
        left: component.x,
        top: component.y,
        borderColor: isSelected ? "#60A5FA" : "rgba(255,255,255,0.2)",
      }}
      onMouseDown={handleMouseDown}
      onClick={handleClick}
    >
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center text-xl font-mono
                     bg-gradient-to-br from-white/10 to-white/5 flex-shrink-0 shadow-lg"
          style={{ color: component.color }}
        >
          {component.icon}
        </div>
        <div className="flex-1">
          <div className="font-medium text-white text-sm">{component.type}</div>
          <div className="text-xs text-gray-400 mt-1">Neural Network Layer</div>
        </div>
      </div>

      {/* Connection ports */}
      <div className="flex justify-between items-center mb-4">
        <button
          className="w-5 h-5 bg-gradient-to-r from-green-400 to-green-500 rounded-full 
                     hover:from-green-300 hover:to-green-400 transition-all duration-200
                     shadow-lg shadow-green-400/30 flex-shrink-0 hover:scale-125 border-2 border-white/20"
          onClick={(e) => {
            e.stopPropagation()
            e.preventDefault()
            onStartConnection(component.id, "input", "input")
          }}
          onMouseDown={(e) => {
            e.stopPropagation()
            e.preventDefault()
          }}
          title="Input Port"
        />
        <div className="text-xs text-gray-500 font-mono">{component.id.toString().slice(-4)}</div>
        <button
          className="w-5 h-5 bg-gradient-to-r from-red-400 to-red-500 rounded-full 
                     hover:from-red-300 hover:to-red-400 transition-all duration-200
                     shadow-lg shadow-red-400/30 flex-shrink-0 hover:scale-125 border-2 border-white/20"
          onClick={(e) => {
            e.stopPropagation()
            e.preventDefault()
            onStartConnection(component.id, "output", "output")
          }}
          onMouseDown={(e) => {
            e.stopPropagation()
            e.preventDefault()
          }}
          title="Output Port"
        />
      </div>

      {/* Parameters preview */}
      <div className="text-xs text-gray-400 space-y-1 bg-white/5 rounded-lg p-2">
        {Object.entries(component.params)
          .slice(0, 3)
          .map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="opacity-70 truncate font-medium">{key}:</span>
              <span className="text-gray-300 truncate max-w-[80px] font-mono">
                {typeof value === "string" ? value.slice(0, 8) : String(value)}
              </span>
            </div>
          ))}
        {Object.keys(component.params).length > 3 && (
          <div className="text-gray-500 text-center font-medium">+{Object.keys(component.params).length - 3} more</div>
        )}
      </div>
    </div>
  )
}
