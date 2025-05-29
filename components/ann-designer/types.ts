export interface Component {
  id: string | number
  type: string
  x: number
  y: number
  params: Record<string, any>
  color: string
  icon: string
  inputs: string[]
  outputs: string[]
}

export interface Connection {
  id: string | number
  from: string | number
  to: string | number
  fromPort: string
  toPort: string
}

export interface ComponentType {
  type: string
  icon: string
  color: string
  params: Record<string, any>
  description?: string
}

export interface ComponentCategory {
  [key: string]: ComponentType[]
}

export interface ModelTemplate {
  id: string
  name: string
  description: string
  category: string
  components: ComponentType[]
  connections: {
    from: number
    to: number
    fromPort: string
    toPort: string
  }[]
}

export interface ModelTemplateCategory {
  [key: string]: ModelTemplate[]
}
