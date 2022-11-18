import { DisplayStory } from '../types'
import Favicon from './favicon'
import { FunctionComponent } from 'react'
import Head from 'next/head'
import StoryCard from './storyCard'

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
      <div className="flex justify-center bg-gray-50 pt-4 lg:pt-8 ">
        <div className="mb-8 w-full px-6 lg:w-4/5">
          <div className="sm:w-1/2 sm:pr-4 lg:w-1/3">
            <StoryCard story={queryStory} />
          </div>
          <div className="space-y-2 pt-8">
            <h2>Similar stories</h2>
            <ul className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {similarStories.map((story) => (
                <li key={story.id} className="mb-4">
                  <StoryCard story={story} />
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </>
  )
}
export default IndexPage
