import { Client, estypes } from '@elastic/elasticsearch'
import { DisplayStory, Story } from '../types'

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

export async function getSimilarStories(
  client: Client,
  queryId: string,
  n: number
): Promise<{ results: DisplayStory[]; took: number }> {
  const queryData = await client
    .get({
      index: process.env.ES_INDEX as string,
      id: queryId,
    })
    .then((res) => res._source as Story)

  const query = {
    knn: {
      field: 'title_embedding',
      query_vector: queryData.title_embedding,
      k: n,
      num_candidates: 100,
    },
  }

  const knnResponse: estypes.KnnSearchResponse<Story> = await client.knnSearch({
    index: process.env.ES_INDEX as string,
    body: query,
  })

  const results = knnResponse.hits.hits.slice(1).map((hit) => parse(hit))
  const took = knnResponse.took
  return { results, took }
}

export function parse(hit: estypes.SearchHit<Story>): DisplayStory {
  return {
    id: hit._id,
    title: hit._source ? hit._source.title : '',
    standfirst: hit._source ? hit._source.standfirst : '',
    thumbnail: hit._source ? hit._source.thumbnail : '',
    url: hit._source ? hit._source.url : '',
  } as DisplayStory
}

export async function getRandomId(client: Client): Promise<string> {
  const randomIdResponse = await client.search({
    index: process.env.ES_INDEX as string,
    body: {
      size: 1,
      query: {
        function_score: {
          random_score: {},
        },
      },
    },
  })
  return randomIdResponse.hits.hits[0]._id
}

export async function getStory(
  client: Client,
  id: string
): Promise<DisplayStory> {
  const storyResponse: estypes.GetResponse<Story> = await client.get({
    index: process.env.ES_INDEX as string,
    id: id,
  })
  return parse(storyResponse) as DisplayStory
}
