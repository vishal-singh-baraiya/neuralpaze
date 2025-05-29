"use client"
import { Button } from "@/components/ui/button"
import { Copy, X, Download } from "lucide-react"

interface CodeModalProps {
  code: string
  onClose: () => void
}

export function CodeModal({ code, onClose }: CodeModalProps) {
  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code)
      alert("Code copied to clipboard!")
    } catch (err) {
      // Fallback for older browsers
      const textarea = document.createElement("textarea")
      textarea.value = code
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand("copy")
      document.body.removeChild(textarea)
      alert("Code copied to clipboard!")
    }
  }

  const downloadCode = () => {
    const blob = new Blob([code], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "generated_model.py"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-black/90 backdrop-blur-xl border border-white/20 rounded-xl w-full max-w-6xl h-[85vh] flex flex-col">
        <div className="p-6 border-b border-white/10 flex-shrink-0">
          <div className="flex justify-between items-center">
            <h3 className="text-xl font-semibold text-white">Generated PyTorch Code</h3>
            <Button
              onClick={onClose}
              variant="ghost"
              size="sm"
              className="text-gray-400 hover:text-white hover:bg-white/10"
            >
              <X size={20} />
            </Button>
          </div>
        </div>

        <div className="flex-1 p-6 overflow-hidden">
          <div className="h-full bg-gray-900/50 rounded-lg border border-white/10 overflow-auto">
            <pre className="p-4 text-sm text-gray-100 whitespace-pre-wrap font-mono leading-relaxed">{code}</pre>
          </div>
        </div>

        <div className="p-6 border-t border-white/10 flex gap-3 flex-shrink-0">
          <Button
            onClick={copyToClipboard}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Copy size={16} />
            Copy to Clipboard
          </Button>
          <Button
            onClick={downloadCode}
            variant="ghost"
            className="flex items-center gap-2 text-green-400 hover:text-green-300 hover:bg-green-500/20 border border-green-500/30"
          >
            <Download size={16} />
            Download File
          </Button>
        </div>
      </div>
    </div>
  )
}
