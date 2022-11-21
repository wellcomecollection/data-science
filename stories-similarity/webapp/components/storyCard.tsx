import { DisplayStory } from '../types'
import { FunctionComponent } from 'react'

type Props = {
  story: DisplayStory
}

function truncate(str: string, n: number) {
  return str.length > n ? str.substr(0, n - 1) + '...' : str
}
const StoryCard: FunctionComponent<Props> = ({ story }) => {
  return (
    <div className="overflow-hidden rounded-lg bg-white shadow">
      <div className="relative pb-2/3">
        <img
          src={story.thumbnail}
          alt={story.title}
          className="absolute h-full w-full object-cover"
        />
      </div>
      <div className="px-6 py-4">
        <h3 className="leading-snug">{story.title}</h3>
        <p className="py-2 text-sm text-gray-700">
          {truncate(story.standfirst, 100)}
        </p>
        <a href={story.url} className="text-sm underline">
          Read this story
        </a>
      </div>
    </div>
  )
}

export default StoryCard
