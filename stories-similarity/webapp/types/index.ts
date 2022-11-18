export type CoreStory = {
  title: string
  standfirst: string
  thumbnail: string
  url: string
}

export type Story = CoreStory & {
  title_embedding: number[]
  standfirst_embedding: number[]
}

export type DisplayStory = CoreStory & { id: string }
