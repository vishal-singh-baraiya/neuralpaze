import type { Component, Connection } from "../types"

export function generatePyTorchCode(components: Component[], connections: Connection[]): string {
  if (!components || components.length === 0) {
    return "# No components to generate code for\n# Add some components to your network first!"
  }

  let code = `import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

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

  code += `class GeneratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
`

  // Generate layer definitions (still need all layers defined)
  components.forEach((comp, index) => {
    const layerName = `layer_${index}_${comp.type.toLowerCase().replace(/[^a-z0-9]/g, "_")}`

    try {
      code += generateLayerDefinition(comp, layerName)
    } catch (error) {
      console.error(`Error processing component ${comp.type}:`, error)
      code += `        # Error processing ${comp.type}: ${error}\n`
    }
  })

  // Generate forward method with proper connection order
  code += generateForwardMethodWithConnections(components, connections)

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
  // Create a map from component id to component
  const componentMap = new Map<string, Component>()
  components.forEach(comp => {
    componentMap.set(comp.id, comp)
  })

  // Build adjacency list and in-degree count
  const adjList = new Map<string, string[]>()
  const inDegree = new Map<string, number>()
  
  // Initialize
  components.forEach(comp => {
    adjList.set(comp.id, [])
    inDegree.set(comp.id, 0)
  })

  // Build the graph from connections
  connections.forEach(conn => {
    if (adjList.has(conn.source) && adjList.has(conn.target)) {
      adjList.get(conn.source)!.push(conn.target)
      inDegree.set(conn.target, (inDegree.get(conn.target) || 0) + 1)
    }
  })

  // Find nodes with no incoming edges (input nodes)
  const queue: string[] = []
  inDegree.forEach((degree, nodeId) => {
    if (degree === 0) {
      queue.push(nodeId)
    }
  })

  const sortedOrder: Component[] = []
  
  while (queue.length > 0) {
    const currentId = queue.shift()!
    const currentComp = componentMap.get(currentId)
    
    if (currentComp) {
      sortedOrder.push(currentComp)
    }

    // Process neighbors
    const neighbors = adjList.get(currentId) || []
    neighbors.forEach(neighborId => {
      const newInDegree = (inDegree.get(neighborId) || 0) - 1
      inDegree.set(neighborId, newInDegree)
      
      if (newInDegree === 0) {
        queue.push(neighborId)
      }
    })
  }

  // If we couldn't sort all components (cycle detected or disconnected components),
  // include remaining components at the end
  const processedIds = new Set(sortedOrder.map(comp => comp.id))
  components.forEach(comp => {
    if (!processedIds.has(comp.id)) {
      sortedOrder.push(comp)
    }
  })

  return sortedOrder
}

function generateForwardMethodWithConnections(components: Component[], connections: Connection[]): string {
  let code = `
    def forward(self, x: torch.Tensor) -> torch.Tensor:
`

  if (components.length === 0) {
    code += "        return x\n"
    return code
  }

  // Get components in topological order
  const sortedComponents = topologicalSort(components, connections)
  
  // Create connection mapping
  const connectionMap = new Map<string, Connection[]>()
  connections.forEach(conn => {
    if (!connectionMap.has(conn.target)) {
      connectionMap.set(conn.target, [])
    }
    connectionMap.get(conn.target)!.push(conn)
  })

  // Track which components have multiple inputs (need special handling)
  const multiInputComponents = new Set<string>()
  connectionMap.forEach((conns, targetId) => {
    if (conns.length > 1) {
      multiInputComponents.add(targetId)
    }
  })

  code += "        # Forward pass through layers following connection order\n"
  code += "        layer_outputs: Dict[str, torch.Tensor] = {}\n"
  code += "        \n"

  // Process components in topological order
  sortedComponents.forEach((comp, orderIndex) => {
    const originalIndex = components.findIndex(c => c.id === comp.id)
    const layerName = `layer_${originalIndex}_${comp.type.toLowerCase().replace(/[^a-z0-9]/g, "_")}`
    const varName = `output_${originalIndex}`
    
    // Determine input for this component
    const inputConnections = connectionMap.get(comp.id) || []
    let inputCode = ""
    
    if (inputConnections.length === 0) {
      // No incoming connections - use original input or previous output
      if (orderIndex === 0) {
        inputCode = "x"
      } else {
        // For disconnected components, use the main flow
        inputCode = "x"
      }
    } else if (inputConnections.length === 1) {
      // Single input connection
      const sourceComp = components.find(c => c.id === inputConnections[0].source)
      if (sourceComp) {
        const sourceIndex = components.findIndex(c => c.id === sourceComp.id)
        inputCode = `layer_outputs.get('${sourceComp.id}', x)`
      } else {
        inputCode = "x"
      }
    } else {
      // Multiple input connections - handle based on component type
      if (comp.type === "Concatenate") {
        const inputTensors = inputConnections.map(conn => {
          const sourceComp = components.find(c => c.id === conn.source)
          if (sourceComp) {
            return `layer_outputs.get('${sourceComp.id}', x)`
          }
          return "x"
        })
        inputCode = `torch.cat([${inputTensors.join(", ")}], dim=-1)`
      } else {
        // For other components with multiple inputs, use the first connection or sum them
        const sourceComp = components.find(c => c.id === inputConnections[0].source)
        if (sourceComp) {
          inputCode = `layer_outputs.get('${sourceComp.id}', x)`
        } else {
          inputCode = "x"
        }
      }
    }

    // Generate the layer execution code
    code += generateLayerExecution(comp, layerName, varName, inputCode, originalIndex)
    
    // Store output for future use
    code += `        layer_outputs['${comp.id}'] = ${varName}\n`
    
    // Update main flow variable
    code += `        x = ${varName}\n`
    code += "        \n"
  })

  code += "        return x\n"
  return code
}

function generateLayerExecution(comp: Component, layerName: string, varName: string, inputCode: string, index: number): string {
  let code = ""

  switch (comp.type) {
    case "MultiHeadAttention":
    case "SparseAttention":
      code += `        ${varName}, _ = self.${layerName}(${inputCode}, ${inputCode}, ${inputCode})  # self-attention\n`
      break

    case "GroupedQueryAttention":
    case "MixtureOfExperts":
    case "MambaBlock":
      code += `        ${varName} = self.${layerName}(${inputCode})\n`
      break

    case "FeedForward":
      code += `        ${varName} = self.${layerName}_linear1(${inputCode})\n`
      code += `        ${varName} = self.${layerName}_activation(${varName})\n`
      if (comp.params.dropout) {
        code += `        ${varName} = self.${layerName}_dropout(${varName})\n`
      }
      code += `        ${varName} = self.${layerName}_linear2(${varName})\n`
      break

    case "GLU":
      code += `        gate = self.${layerName}_gate(${inputCode})\n`
      code += `        up = self.${layerName}_up(${inputCode})\n`
      code += `        ${varName} = self.${layerName}_down(gate * F.${comp.params.activation || "silu"}(up))\n`
      break

    case "RotaryPositionalEncoding":
      code += `        pos_emb = self.${layerName}(${inputCode})\n`
      code += `        ${varName} = ${inputCode} + pos_emb\n`
      break

    case "ALiBi":
      code += `        # ALiBi bias applied during attention (placeholder)\n`
      code += `        ${varName} = ${inputCode}  # ALiBi modifies attention scores, not input directly\n`
      break

    case "Residual":
      // For residual connections, we need to handle the skip connection
      code += `        # Residual connection\n`
      code += `        ${varName} = ${inputCode}  # Main path (skip connection should be handled by graph structure)\n`
      break

    case "Concatenate":
      // Input code already handles concatenation for this type
      code += `        ${varName} = ${inputCode}\n`
      break

    case "Split":
      code += `        # Split tensor (using first output for now)\n`
      const splitDim = comp.params.dim || -1
      const numSplits = comp.params.num_splits || 2
      code += `        split_outputs = torch.split(${inputCode}, ${inputCode}.size(${splitDim}) // ${numSplits}, dim=${splitDim})\n`
      code += `        ${varName} = split_outputs[0]  # Using first split\n`
      break

    case "Add":
      // For add operation with multiple inputs, input code should handle it
      code += `        ${varName} = ${inputCode}\n`
      break

    default:
      code += `        ${varName} = self.${layerName}(${inputCode})\n`
  }

  return code
}

// Keep the existing helper functions unchanged
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
