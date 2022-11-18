import { Client, estypes } from '@elastic/elasticsearch'

export type CoreDocument = {
  title: string
  standfirst: string
  thumbnail: string
  url: string
}
export type Document = CoreDocument & {
  title_embedding: number[]
  standfirst_embedding: number[]
}
export type DisplayDocument = CoreDocument & { id: string }

let client: Client
export function getClient(): Client {
  client = new Client({
    node: process.env.ES_HOST as string,
    auth: {
      username: process.env.ES_USERNAME as string,
      password: process.env.ES_PASSWORD as string,
    },
  })
  return client
}

export async function similarity(
  client: Client,
  queryId: string,
  n: number
): Promise<{ results: DisplayDocument[]; took: number }> {
  const queryData = await client
    .get({
      index: process.env.ES_INDEX as string,
      id: queryId,
    })
    .then((res) => res._source as Document)

  const query = {
    knn: {
      field: 'title_embedding',
      query_vector: queryData.title_embedding,
      k: n,
      num_candidates: 100,
    },
  }

  const knnResponse: estypes.KnnSearchResponse<Document> =
    await client.knnSearch({
      index: process.env.ES_INDEX as string,
      body: query,
    })

  const results = knnResponse.hits.hits.map(
    (hit) =>
      ({
        id: hit._id,
        title: hit._source ? hit._source.title : '',
        standfirst: hit._source ? hit._source.standfirst : '',
        thumbnail: hit._source ? hit._source.thumbnail : '',
        url: hit._source ? hit._source.url : '',
      } as DisplayDocument)
  )
  const took = knnResponse.took
  return { results, took }
}
