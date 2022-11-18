import { GetServerSideProps, NextPage } from 'next'
import {
  getClient,
  getRandomId,
  getSimilarStories,
  getStory,
} from '../services'

import { DisplayStory } from '../types'
import IndexPage from '../components/indexPage'

type Props = {
  queryStory: DisplayStory
  similarStories: DisplayStory[]
}
const Index: NextPage<Props> = ({ queryStory, similarStories }) => {
  return <IndexPage queryStory={queryStory} similarStories={similarStories} />
}

export const getServerSideProps: GetServerSideProps = async () => {
  const client = await getClient()
  const queryId = await getRandomId(client)
  const queryStory = await getStory(client, queryId)
  const similarStories = await getSimilarStories(client, queryId, 10).then(
    (res) => res.results
  )
  return { props: { queryStory, similarStories } }
}

export default Index
