"use client"

export function ConnectionIndicator() {
  return (
    <div className="fixed top-6 left-1/2 transform -translate-x-1/2 z-50">
      <div className="bg-blue-500/90 backdrop-blur-sm text-white px-6 py-3 rounded-lg border border-blue-400/30 shadow-lg">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
          <span className="text-sm font-medium">Click on another component to connect</span>
        </div>
      </div>
    </div>
  )
}
