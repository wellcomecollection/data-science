import {
  DisplayStory,
  EmbeddingField,
  EmbeddingFields,
  embeddingFields,
} from '../types'
import { FunctionComponent, useCallback, useState } from 'react'

import Favicon from './favicon'
import Head from 'next/head'
import StoryCard from './storyCard'

type Props = {
  queryStory: DisplayStory
  similarStories: DisplayStory[]
  field: EmbeddingField
}

const IndexPage: FunctionComponent<Props> = ({
  queryStory,
  similarStories,
  field,
}) => {
  const [selectedQueryfield, setQueryField] = useState<EmbeddingField>(field)

  const handleChange = useCallback((e) => {
    setQueryField(e.target.value as EmbeddingField)
    const url = new URL(window.location.href)
    url.searchParams.set('field', e.target.value as string)
    window.history.pushState({}, '', url.toString())
    window.location.reload()
  }, [])

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

      <div className="flex justify-center bg-gray-50 pt-4 lg:pt-8">
        <div className="mb-8 w-full px-6 lg:w-4/5">
          <div className="sm:w-1/2 sm:pr-4 lg:w-1/3">
            <StoryCard story={queryStory} />
          </div>
          <div className="space-y-4 pt-8">
            <h2>Similar stories</h2>
            <div className="flex flex-col sm:flex-row sm:space-x-2">
              <p>Select query field:</p>
              <select
                className="font-mono text-sm md:ml-2"
                onChange={handleChange}
                value={selectedQueryfield}
              >
                {embeddingFields.map((fieldName: EmbeddingField) => (
                  <option key={fieldName} value={fieldName}>
                    {fieldName}
                  </option>
                ))}
              </select>
            </div>
            <ul className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {similarStories.map((story) => (
                <li key={story.id} className="mb-4">
                  <a href={`/${story.id}${field ? `?field=${field}` : ''}`}>
                    <StoryCard story={story} />
                  </a>
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
