import { DisplayStory } from '../types'
import Favicon from './favicon'
import { FunctionComponent } from 'react'
import Head from 'next/head'

type Props = {
  queryStory: DisplayStory
  similarStories: DisplayStory[]
}
const IndexPage: FunctionComponent<Props> = ({
  queryStory,
  similarStories,
}) => {
  return (
    <>
      <Head>
        <title>Stories similarity</title>
        <meta
          name="description"
          content="Prototype UI for surfacing similar stories based on inferred text embeddings"
        />
        <Favicon emoji="📝" />
      </Head>
      <div>query story id: {queryStory.id}</div>
      <div>similar stories:</div>
      <ul>
        {similarStories.map((story) => (
          <li key={story.id}>{story.id}</li>
        ))}
      </ul>
    </>
  )
}
export default IndexPage
