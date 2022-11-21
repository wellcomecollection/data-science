import { DisplayStory, EmbeddingField } from '../types'
import { GetServerSideProps, NextPage } from 'next'
import { getClient, getSimilarStories, getStory } from '../services'

import IndexPage from '../components/indexPage'

type Props = {
  queryStory: DisplayStory
  similarStories: DisplayStory[]
  field: EmbeddingField
}
const Index: NextPage<Props> = ({ queryStory, similarStories, field }) => {
  return (
    <IndexPage
      queryStory={queryStory}
      similarStories={similarStories}
      field={field}
    />
  )
}

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  const queryId = query.id as string
  const field = query.field
    ? (query.field as EmbeddingField)
    : 'title_embedding'
  const client = await getClient()
  const queryStory = await getStory(client, queryId)
  const similarStories = await getSimilarStories(
    client,
    queryId,
    10,
    field
  ).then((res) => res.results)
  return { props: { queryStory, similarStories, field } }
}

export default Index
