"use client"
import type { Component } from "./types"

interface ConnectionLineProps {
  from: Component
  to: Component
}

export function ConnectionLine({ from, to }: ConnectionLineProps) {
  const fromX = from.x + 220 // Right side of component
  const fromY = from.y + 50 // Middle of component
  const toX = to.x // Left side of component
  const toY = to.y + 50 // Middle of component

  // Create a smooth curved path
  const midX = (fromX + toX) / 2
  const controlX1 = fromX + Math.min(100, Math.abs(toX - fromX) * 0.5)
  const controlX2 = toX - Math.min(100, Math.abs(toX - fromX) * 0.5)

  const path = `M ${fromX} ${fromY} C ${controlX1} ${fromY}, ${controlX2} ${toY}, ${toX} ${toY}`

  return (
    <g>
      {/* Glow effect */}
      <path d={path} stroke="rgba(96, 165, 250, 0.3)" strokeWidth="6" fill="none" className="blur-sm" />
      {/* Main line */}
      <path
        d={path}
        stroke="rgba(96, 165, 250, 0.8)"
        strokeWidth="3"
        fill="none"
        markerEnd="url(#arrowhead)"
        className="drop-shadow-sm"
      />
      {/* Animated pulse */}
      <path d={path} stroke="rgba(96, 165, 250, 1)" strokeWidth="1" fill="none" className="animate-pulse" />
    </g>
  )
}
