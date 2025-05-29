import type { Component, Connection } from "../types"

export function generatePyTorchCode(components: Component[], connections: Connection[]): string {
  if (!components || components.length === 0) {
    return generateEmptyNetworkCode()
  }

  try {
    let code = generateImports()
    code += generateHelperClasses(components)
    code += generateMainNetworkClass(components, connections)
    code += generateUsageExample()

    return code
  } catch (error) {
    console.error("Code generation error:", error)
    return generateErrorCode(error)
  }
}

function generateEmptyNetworkCode(): string {
  return `import torch
import torch.nn as nn
import torch.nn.functional as F

class EmptyNetwork(nn.Module):
    """
    Empty neural network - add some components to generate meaningful code!
    """
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.placeholder(x)

# Usage:
# model = EmptyNetwork()
# x = torch.randn(1, 10)  # Example input
# output = model(x)
# print(f"Output shape: {output.shape}")
`
}

function generateErrorCode(error: any): string {
  return `# Error generating PyTorch code: ${error}
# Please check your network configuration and try again.

import torch
import torch.nn as nn

class ErrorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_layer = nn.Identity()
    
    def forward(self, x):
        print("Network generation failed - using identity function")
        return self.error_layer(x)
`
}

function generateImports(): string {
  return `import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

`
}

function generateHelperClasses(components: Component[]): string {
  const usedTypes = new Set(components.map((c) => c.type))
  let code = ""

  if (usedTypes.has("CustomLayer")) {
    const customLayers = components.filter((c) => c.type === "CustomLayer")
    customLayers.forEach((comp) => {
      if (comp.params.name && comp.params.code) {
        code += `
class ${comp.params.name}(nn.Module):
    def __init__(self):
        super().__init__()
        ${comp.params.code
          .split("\n")
          .map((line) => (line.trim() ? `        ${line}` : ""))
          .join("\n")}

`
      }
    })
  }

  return code
}

function generateMainNetworkClass(components: Component[], connections: Connection[]): string {
  let code = `
class GeneratedNetwork(nn.Module):
    """
    Generated Neural Network
    
    Architecture Summary:
    - Total layers: ${components.length}
    - Total connections: ${connections.length}
    """
    
    def __init__(self):
        super().__init__()
        
        # Network layers
`

  // Generate layer definitions
  components.forEach((comp, index) => {
    const layerName = `layer_${index}_${comp.type.toLowerCase().replace(/[^a-z0-9]/g, "_")}`
    code += generateLayerDefinition(comp, layerName)
  })

  // Generate forward method
  code += generateForwardMethod(components)

  return code
}

function generateLayerDefinition(comp: Component, layerName: string): string {
  const params = comp.params || {}

  switch (comp.type) {
    case "Linear":
      return `        self.${layerName} = nn.Linear(${params.input_size || 512}, ${params.output_size || 512})\n`

    case "Conv2D":
      return `        self.${layerName} = nn.Conv2d(${params.in_channels || 3}, ${params.out_channels || 64}, ${params.kernel_size || 3}, padding=${params.padding || 1})\n`

    case "ReLU":
      return `        self.${layerName} = nn.ReLU()\n`

    case "GELU":
      return `        self.${layerName} = nn.GELU()\n`

    case "Dropout":
      return `        self.${layerName} = nn.Dropout(${params.p || 0.1})\n`

    case "LayerNorm":
      return `        self.${layerName} = nn.LayerNorm(${params.normalized_shape || 512})\n`

    case "MultiHeadAttention":
      return `        self.${layerName} = nn.MultiheadAttention(${params.d_model || 512}, ${params.num_heads || 8}, batch_first=True)\n`

    case "CustomLayer":
      if (params.name) {
        return `        self.${layerName} = ${params.name}()\n`
      }
      return `        self.${layerName} = nn.Identity()  # Custom layer not configured\n`

    default:
      return `        self.${layerName} = nn.Identity()  # ${comp.type}\n`
  }
}

function generateForwardMethod(components: Component[]): string {
  let code = `
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
`

  if (components.length === 0) {
    code += "        return x\n"
    return code
  }

  components.forEach((comp, index) => {
    const layerName = `layer_${index}_${comp.type.toLowerCase().replace(/[^a-z0-9]/g, "_")}`

    if (comp.type === "MultiHeadAttention") {
      code += `        x, _ = self.${layerName}(x, x, x)  # Self-attention\n`
    } else {
      code += `        x = self.${layerName}(x)\n`
    }
  })

  code += "        return x\n"
  return code
}

function generateUsageExample(): string {
  return `

# Usage Example
if __name__ == "__main__":
    # Create model
    model = GeneratedNetwork()
    
    # Print model info
    print("Model Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_dim = 512
    
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")
    
    try:
        with torch.no_grad():
            output = model(x)
        print(f"Output shape: {output.shape}")
        print("✅ Forward pass successful!")
    except Exception as e:
        print(f"❌ Error: {e}")
`
}
