import { Client, estypes } from '@elastic/elasticsearch'

import { SearchTotalHits } from '@elastic/elasticsearch/lib/api/types'
import blankQuery from '../../data/queries/search-as-you-type.json'

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
    node: process.env.ES_PROTOTYPE_HOST,
    auth: {
      username: process.env.ES_PROTOTYPE_USERNAME,
      password: process.env.ES_PROTOTYPE_PASSWORD,
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
    index: process.env.INDEX_NAME,
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
