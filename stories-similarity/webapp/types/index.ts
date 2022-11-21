export type CoreStory = {
  title: string
  standfirst: string
  thumbnail: string
  url: string
}

export type EmbeddingFields = {
  title_embedding: number[]
  standfirst_embedding: number[]
  shared_embedding_max: number[]
  shared_embedding_concat: number[]
}

export type Story = CoreStory & EmbeddingFields

export type DisplayStory = CoreStory & { id: string }

export const embeddingFields = [
  'title_embedding',
  'standfirst_embedding',
  'shared_embedding_max',
  'shared_embedding_concat',
]

export type EmbeddingField = typeof embeddingFields[number]
