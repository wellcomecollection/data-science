import { Client, estypes } from '@elastic/elasticsearch'

import { SearchTotalHits } from '@elastic/elasticsearch/lib/api/types'
import blankQuery from '../queries/search-as-you-type.json'

export type Document = {
  type: string
  title: string
  contributors: string
  url: string
  image?: string
}
export type Result = { id: string } & Document

let client: Client
export function getClient(): Client {
  client = new Client({
    node: process.env.ES_PROTOTYPE_HOST as string,
    auth: {
      username: process.env.ES_PROTOTYPE_USERNAME as string,
      password: process.env.ES_PROTOTYPE_PASSWORD as string,
    },
  })
  return client
}

export async function search(
  client: Client,
  searchTerms: string,
  n: number
): Promise<{ results: Result[]; total: number; took: number }> {
  const query = JSON.parse(
    JSON.stringify(blankQuery).replace(/{{query}}/g, searchTerms)
  )
  const searchResponse: estypes.SearchResponse<Document> = await client.search({
    index: process.env.INDEX_NAME as string,
    body: query,
    size: n,
  })

  const results = searchResponse.hits.hits.map(
    (hit) => ({ id: hit._id, ...hit._source } as Result)
  )
  const total = (searchResponse.hits.total as SearchTotalHits).value
  const took = searchResponse.took
  return { results, total, took }
}
