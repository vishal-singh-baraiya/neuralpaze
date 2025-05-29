import type { Component, Connection } from "../types"

export function generatePyTorchCode(components: Component[], connections: Connection[]): string {
  if (!components || components.length === 0) {
    return "# No components to generate code for\n# Add some components to your network first!"
  }

  let code = `import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

`

  // Check which advanced components are actually used
  const usedComponentTypes = new Set(components.map((comp) => comp.type))

  // Generate custom component classes first
  const customComponents = components.filter((comp) => comp.type === "CustomLayer")

  customComponents.forEach((comp) => {
    if (comp.params.code && comp.params.code.trim()) {
      code += `class ${comp.params.name || "CustomLayer"}(nn.Module):
    def __init__(self):
        super().__init__()
        ${comp.params.code
          .split("\n")
          .map((line) => (line.trim() ? `        ${line}` : ""))
          .join("\n")}

`
    }
  })

  // Only add helper classes that are actually used
  const helperClasses = generateHelperClasses(usedComponentTypes)
  if (helperClasses.trim()) {
    code += helperClasses
  }

  // Sort components based on their connections to get execution order
  const sortedComponents = topologicalSort(components, connections)
  const connectionInfo = analyzeConnections(components, connections)

  // Check if the architecture is sequential
  const isSequential = isSequentialArchitecture(sortedComponents, connectionInfo)

  code += `class GeneratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
`

  // Generate layer definitions in the execution order
  sortedComponents.forEach((comp, executionIndex) => {
    const layerName = `layer_${executionIndex}_${comp.type.toLowerCase().replace(/[^a-z0-9]/g, "_")}`
    
    try {
      code += generateLayerDefinition(comp, layerName)
    } catch (error) {
      console.error(`Error processing component ${comp.type}:`, error)
      code += `        # Error processing ${comp.type}: ${error}\n`
    }
  })

  // Generate forward method
  code += generateForwardMethodWithConnections(sortedComponents, connectionInfo, isSequential)

  code += `
# Usage example:
# model = GeneratedNetwork()
# print(model)
# 
# # For training:
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()
`

  return code
}

function topologicalSort(components: Component[], connections: Connection[]): Component[] {
  // Create adjacency list and in-degree count
  const adjacencyList = new Map<string, string[]>()
  const inDegree = new Map<string, number>()
  const componentMap = new Map<string, Component>()

  // Initialize maps
  components.forEach(comp => {
    adjacencyList.set(comp.id, [])
    inDegree.set(comp.id, 0)
    componentMap.set(comp.id, comp)
  })

  // Build graph from connections
  connections.forEach(conn => {
    if (adjacencyList.has(conn.from) && adjacencyList.has(conn.to)) {
      adjacencyList.get(conn.from)!.push(conn.to)
      inDegree.set(conn.to, (inDegree.get(conn.to) || 0) + 1)
    }
  })

  // Kahn's algorithm for topological sorting
  const queue: string[] = []
  const result: Component[] = []

  // Find all nodes with no incoming edges
  inDegree.forEach((degree, nodeId) => {
    if (degree === 0) {
      queue.push(nodeId)
    }
  })

  while (queue.length > 0) {
    const currentId = queue.shift()!
    const currentComponent = componentMap.get(currentId)!
    result.push(currentComponent)

    // Process all neighbors
    adjacencyList.get(currentId)!.forEach(neighborId => {
      inDegree.set(neighborId, inDegree.get(neighborId)! - 1)
      if (inDegree.get(neighborId) === 0) {
        queue.push(neighborId)
      }
    })
  }

  // If we couldn't sort all components, there might be a cycle or disconnected components
  if (result.length !== components.length) {
    console.warn("Could not perform complete topological sort - using original order for missing components")
    // Add any missing components at the end
    const resultIds = new Set(result.map(c => c.id))
    components.forEach(comp => {
      if (!resultIds.has(comp.id)) {
        result.push(comp)
      }
    })
  }

  return result
}

function analyzeConnections(components: Component[], connections: Connection[]) {
  const connectionMap = new Map<string, { inputs: string[], outputs: string[] }>()
  
  // Initialize connection info for each component
  components.forEach(comp => {
    connectionMap.set(comp.id, { inputs: [], outputs: [] })
  })

  // Build connection mappings
  connections.forEach(conn => {
    const fromInfo = connectionMap.get(conn.from)
    const toInfo = connectionMap.get(conn.to)
    
    if (fromInfo) fromInfo.outputs.push(conn.to)
    if (toInfo) toInfo.inputs.push(conn.from)
  })

  return connectionMap
}

function isSequentialArchitecture(
  sortedComponents: Component[],
  connectionInfo: Map<string, { inputs: string[], outputs: string[] }>
): boolean {
  // Check if the architecture is strictly sequential:
  // - Each component (except first) has exactly one input
  // - Each component (except last) has exactly one output
  // - The components form a single chain (topological sort matches connection order)
  for (let i = 0; i < sortedComponents.length; i++) {
    const comp = sortedComponents[i]
    const compInfo = connectionInfo.get(comp.id)!
    
    // First component: no inputs, exactly one output
    if (i === 0) {
      if (compInfo.inputs.length > 0 || compInfo.outputs.length !== 1) {
        return false
      }
    }
    // Last component: exactly one input, no outputs
    else if (i === sortedComponents.length - 1) {
      if (compInfo.inputs.length !== 1 || compInfo.outputs.length > 0) {
        return false
      }
    }
    // Middle components: exactly one input, one output
    else {
      if (compInfo.inputs.length !== 1 || compInfo.outputs.length !== 1) {
        return false
      }
    }
    
    // Check if the output of component i connects to the input of component i+1
    if (i < sortedComponents.length - 1) {
      const nextComp = sortedComponents[i + 1]
      if (!compInfo.outputs.includes(nextComp.id)) {
        return false
      }
    }
  }
  return true
}

function generateForwardMethodWithConnections(
  sortedComponents: Component[], 
  connectionInfo: Map<string, { inputs: string[], outputs: string[] }>,
  isSequential: boolean
): string {
  let code = `
    def forward(self, x: torch.Tensor) -> torch.Tensor:
`

  if (sortedComponents.length === 0) {
    code += "        return x\n"
    return code
  }

  // Create a mapping of component IDs to layer names
  const idToLayerName = new Map<string, string>()
  sortedComponents.forEach((comp, executionIndex) => {
    const layerName = `layer_${executionIndex}_${comp.type.toLowerCase().replace(/[^a-z0-9]/g, "_")}`
    idToLayerName.set(comp.id, layerName)
  })

  if (isSequential) {
    code += `        # Forward pass through layers\n`
    sortedComponents.forEach((comp, executionIndex) => {
      const layerName = idToLayerName.get(comp.id)!
      code += `        x = self.${layerName}(x)\n`
    })
    code += `        return x\n`
  } else {
    // Fallback to dictionary-based approach for non-sequential architectures
    code += `        # Forward pass following the network connections\n`
    code += `        activations: Dict[str, torch.Tensor] = {}\n`
    code += `        \n`

    // Find input components
    const inputComponents = sortedComponents.filter(comp => {
      const info = connectionInfo.get(comp.id)
      return !info || info.inputs.length === 0
    })

    // Initialize input activations
    if (inputComponents.length === 1) {
      code += `        # Initialize input\n`
      code += `        activations['${idToLayerName.get(inputComponents[0].id)}'] = x\n`
      code += `        \n`
    } else if (inputComponents.length > 1) {
      code += `        # Multiple input components detected\n`
      code += `        # Assuming input x will be used for the first component\n`
      inputComponents.forEach((comp, idx) => {
        if (idx === 0) {
          code += `        activations['${idToLayerName.get(comp.id)}'] = x\n`
        } else {
          code += `        # activations['${idToLayerName.get(comp.id)}'] = your_input_${idx}  # TODO: Provide appropriate input\n`
        }
      })
      code += `        \n`
    } else {
      code += `        # No clear input component found, using first component\n`
      code += `        activations['${idToLayerName.get(sortedComponents[0].id)}'] = x\n`
      code += `        \n`
    }

    // Process components in execution order
    sortedComponents.forEach((comp, executionIndex) => {
      const layerName = idToLayerName.get(comp.id)!
      const compInfo = connectionInfo.get(comp.id)!
      
      // Skip if this is an input component that's already initialized
      if (inputComponents.includes(comp) && compInfo.inputs.length === 0) {
        return
      }
      
      // Determine input for this component
      let inputExpression = ""
      if (compInfo.inputs.length === 0) {
        if (executionIndex === 0) {
          inputExpression = "x"
        } else {
          const prevComp = sortedComponents[executionIndex - 1]
          inputExpression = `activations['${idToLayerName.get(prevComp.id)}']`
        }
      } else if (compInfo.inputs.length === 1) {
        inputExpression = `activations['${idToLayerName.get(compInfo.inputs[0])}']`
      } else {
        if (comp.type === "Concatenate") {
          const inputRefs = compInfo.inputs.map(id => `activations['${idToLayerName.get(id)}']`).join(", ")
          inputExpression = `torch.cat([${inputRefs}], dim=-1)`
        } else if (comp.type === "Add" || comp.type === "Residual") {
          const inputRefs = compInfo.inputs.map(id => `activations['${idToLayerName.get(id)}']`).join(" + ")
          inputExpression = inputRefs
        } else {
          inputExpression = `activations['${idToLayerName.get(compInfo.inputs[0])}']`
          code += `        # Warning: ${comp.type} received multiple inputs, using first one\n`
        }
      }

      // Generate the forward pass for this component
      code += `        # Step ${executionIndex}: ${comp.type} (${layerName})\n`
      
      switch (comp.type) {
        case "MultiHeadAttention":
        case "SparseAttention":
          code += `        activations['${layerName}'], _ = self.${layerName}(${inputExpression}, ${inputExpression}, ${inputExpression})  # self-attention\n`
          break

        case "GroupedQueryAttention":
        case "MixtureOfExperts":
        case "MambaBlock":
          code += `        activations['${layerName}'] = self.${layerName}(${inputExpression})\n`
          break

        case "FeedForward":
          code += `        temp = self.${layerName}_linear1(${inputExpression})\n`
          code += `        temp = self.${layerName}_activation(temp)\n`
          if (comp.params.dropout) {
            code += `        temp = self.${layerName}_dropout(temp)\n`
          }
          code += `        activations['${layerName}'] = self.${layerName}_linear2(temp)\n`
          break

        case "GLU":
          code += `        gate = self.${layerName}_gate(${inputExpression})\n`
          code += `        up = self.${layerName}_up(${inputExpression})\n`
          code += `        activations['${layerName}'] = self.${layerName}_down(gate * F.${comp.params.activation || "silu"}(up))\n`
          break

        case "RotaryPositionalEncoding":
          code += `        pos_emb = self.${layerName}(${inputExpression})\n`
          code += `        activations['${layerName}'] = ${inputExpression} + pos_emb\n`
          break

        case "ALiBi":
          code += `        # ALiBi bias applied during attention (placeholder)\n`
          code += `        activations['${layerName}'] = ${inputExpression}  # ALiBi modifies attention scores, not input directly\n`
          break

        case "Residual":
          if (compInfo.inputs.length >= 2) {
            code += `        activations['${layerName}'] = ${inputExpression}\n`
          } else {
            code += `        # Residual connection needs at least 2 inputs\n`
            code += `        activations['${layerName}'] = ${inputExpression}\n`
          }
          break

        case "Add":
          if (compInfo.inputs.length >= 2) {
            code += `        activations['${layerName}'] = ${inputExpression}\n`
          } else {
            code += `        activations['${layerName}'] = ${inputExpression}\n`
          }
          break

        case "Concatenate":
          if (compInfo.inputs.length >= 2) {
            code += `        activations['${layerName}'] = ${inputExpression}\n`
          } else {
            code += `        activations['${layerName}'] = ${inputExpression}\n`
          }
          break

        case "Split":
          const outputsCount = compInfo.outputs.length || 2
          code += `        split_outputs = torch.chunk(${inputExpression}, ${outputsCount}, dim=-1)\n`
          code += `        activations['${layerName}'] = split_outputs[0]  # Using first split as main output\n`
          break

        default:
          code += `        activations['${layerName}'] = self.${layerName}(${inputExpression})\n`
      }
      
      code += `        \n`
    })

    // Find output components
    const outputComponents = sortedComponents.filter(comp => {
      const info = connectionInfo.get(comp.id)
      return !info || info.outputs.length === 0
    })

    if (outputComponents.length === 1) {
      code += `        # Return final output\n`
      code += `        return activations['${idToLayerName.get(outputComponents[0].id)}']\n`
    } else if (outputComponents.length > 1) {
      code += `        # Multiple output components detected, returning tuple\n`
      const outputRefs = outputComponents.map(comp => `activations['${idToLayerName.get(comp.id)}']`).join(", ")
      code += `        return (${outputRefs})\n`
    } else {
      code += `        # No clear output component, returning last activation\n`
      if (sortedComponents.length > 0) {
        code += `        return activations['${idToLayerName.get(sortedComponents[sortedComponents.length - 1].id)}']\n`
      } else {
        code += `        return x\n`
      }
    }
  }

  return code
}

function generateHelperClasses(usedComponentTypes: Set<string>): string {
  let code = ""

  if (usedComponentTypes.has("RotaryPositionalEncoding")) {
    code += `
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

`
  }

  if (usedComponentTypes.has("ALiBi")) {
    code += `
class ALiBi(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
        
    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        seq_len = attention_scores.size(-1)
        bias = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        bias = bias.abs() * -1
        bias = bias.unsqueeze(0) * self.slopes.view(-1, 1, 1)
        return attention_scores + bias

`
  }

  if (usedComponentTypes.has("MixtureOfExperts")) {
    code += `
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, d_ff: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        gate_scores = self.gate(x_flat)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_score = top_k_scores[:, i].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_output = self.experts[expert_id](x_flat[mask])
                    output[mask] += expert_score[mask] * expert_output
                    
        return output.view(batch_size, seq_len, d_model)

`
  }

  if (usedComponentTypes.has("GroupedQueryAttention")) {
    code += `
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Repeat k,v for grouped attention
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)

`
  }

  if (usedComponentTypes.has("MambaBlock")) {
    code += `
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(self.d_model * self.expand, dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # (B, L, D)
        
        # SSM
        x = F.silu(x)
        
        # Simplified SSM computation (actual Mamba is more complex)
        y = x * F.sigmoid(self.dt_proj(x))
        
        # Output projection
        y = y * F.silu(res)
        return self.out_proj(y)

`
  }

  return code
}

function generateLayerDefinition(comp: Component, layerName: string): string {
  let code = ""

  switch (comp.type) {
    case "CustomLayer":
      code += `        self.${layerName} = ${comp.params.name || "CustomLayer"}()\n`
      break

    case "Linear":
      code += `        self.${layerName} = nn.Linear(${comp.params.input_size || 512}, ${comp.params.output_size || 512}`
      if (comp.params.bias !== undefined) code += `, bias=${comp.params.bias}`
      code += ")\n"
      break

    case "Conv1D":
      code += `        self.${layerName} = nn.Conv1d(${comp.params.in_channels || 64}, ${comp.params.out_channels || 64}, ${comp.params.kernel_size || 3}`
      if (comp.params.stride) code += `, stride=${comp.params.stride}`
      if (comp.params.padding) code += `, padding=${comp.params.padding}`
      code += ")\n"
      break

    case "Conv2D":
      code += `        self.${layerName} = nn.Conv2d(${comp.params.in_channels || 3}, ${comp.params.out_channels || 64}, ${comp.params.kernel_size || 3}`
      if (comp.params.stride) code += `, stride=${comp.params.stride}`
      if (comp.params.padding) code += `, padding=${comp.params.padding}`
      code += ")\n"
      break

    case "Embedding":
      code += `        self.${layerName} = nn.Embedding(${comp.params.vocab_size || 50000}, ${comp.params.embed_dim || 512})\n`
      break

    case "MultiHeadAttention":
      code += `        self.${layerName} = nn.MultiheadAttention(embed_dim=${comp.params.d_model || 512}, num_heads=${comp.params.num_heads || 8}`
      if (comp.params.dropout) code += `, dropout=${comp.params.dropout}`
      code += ")\n"
      break

    case "GroupedQueryAttention":
      code += `        self.${layerName} = GroupedQueryAttention(d_model=${comp.params.d_model || 512}, num_heads=${comp.params.num_heads || 8}, num_kv_heads=${comp.params.num_kv_heads || 2})\n`
      break

    case "SparseAttention":
      code += `        self.${layerName} = nn.MultiheadAttention(embed_dim=${comp.params.d_model || 512}, num_heads=${comp.params.num_heads || 8})\n`
      code += `        # Note: Sparse attention pattern not fully implemented in this example\n`
      break

    case "MixtureOfExperts":
      code += `        self.${layerName} = MixtureOfExperts(d_model=${comp.params.d_model || 512}, num_experts=${comp.params.num_experts || 8}, top_k=${comp.params.top_k || 2}, d_ff=${comp.params.d_ff || 2048})\n`
      break

    case "MambaBlock":
      code += `        self.${layerName} = MambaBlock(d_model=${comp.params.d_model || 512}, d_state=${comp.params.d_state || 16}, d_conv=${comp.params.d_conv || 4}, expand=${comp.params.expand || 2})\n`
      break

    case "RotaryPositionalEncoding":
      code += `        self.${layerName} = RotaryPositionalEncoding(d_model=${comp.params.d_model || 512}, max_len=${comp.params.max_len || 8192})\n`
      break

    case "ALiBi":
      code += `        self.${layerName} = ALiBi(num_heads=${comp.params.num_heads || 8})\n`
      break

    case "TransformerBlock":
    case "GPTBlock":
    case "LlamaBlock":
      code += `        self.${layerName} = nn.TransformerEncoderLayer(d_model=${comp.params.d_model || 512}, nhead=${comp.params.num_heads || 8}`
      if (comp.params.d_ff) code += `, dim_feedforward=${comp.params.d_ff}`
      if (comp.params.dropout) code += `, dropout=${comp.params.dropout}`
      code += ")\n"
      break

    case "LayerNorm":
      code += `        self.${layerName} = nn.LayerNorm(${comp.params.normalized_shape || 512})\n`
      break

    case "BatchNorm":
      code += `        self.${layerName} = nn.BatchNorm1d(${comp.params.num_features || 512})\n`
      break

    case "Dropout":
      code += `        self.${layerName} = nn.Dropout(${comp.params.p || 0.1})\n`
      break

    case "ReLU":
    case "GELU":
    case "SiLU":
    case "Tanh":
      code += `        self.${layerName} = nn.${comp.type}()\n`
      break

    case "Softmax":
      code += `        self.${layerName} = nn.Softmax(dim=${comp.params.dim || -1})\n`
      break

    case "LeakyReLU":
      code += `        self.${layerName} = nn.LeakyReLU(negative_slope=${comp.params.negative_slope || 0.01})\n`
      break

    case "FeedForward":
      code += `        self.${layerName}_linear1 = nn.Linear(${comp.params.d_model || 512}, ${comp.params.d_ff || 2048})\n`
      code += `        self.${layerName}_activation = nn.${comp.params.activation?.toUpperCase() || "GELU"}()\n`
      code += `        self.${layerName}_linear2 = nn.Linear(${comp.params.d_ff || 2048}, ${comp.params.d_model || 512})\n`
      if (comp.params.dropout) {
        code += `        self.${layerName}_dropout = nn.Dropout(${comp.params.dropout})\n`
      }
      break

    case "GLU":
      code += `        self.${layerName}_gate = nn.Linear(${comp.params.d_model || 512}, ${comp.params.d_ff || 2048})\n`
      code += `        self.${layerName}_up = nn.Linear(${comp.params.d_model || 512}, ${comp.params.d_ff || 2048})\n`
      code += `        self.${layerName}_down = nn.Linear(${comp.params.d_ff || 2048}, ${comp.params.d_model || 512})\n`
      break

    default:
      const params = Object.entries(comp.params || {})
        .filter(([k, v]) => v !== null && v !== undefined && v !== "")
        .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
        .join(", ")
      code += `        self.${layerName} = nn.${comp.type}(${params})\n`
  }

  return code
}
