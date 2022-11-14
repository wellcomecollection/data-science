import { FunctionComponent } from 'react'
import Image from 'next/image'
import { Result } from '../services/search'

type Props = {
  result: Result
}

const SearchResult: FunctionComponent<Props> = ({ result }) => {
  return (
    <a href={result.url} className="flex py-4 px-2 hover:bg-gray-100">
      {result.image && (
        <div className="relative mr-4 w-40 flex-none">
          <Image
            src={result.image}
            alt=""
            layout="fill"
            objectFit="cover"
            className="relative rounded-md"
          />
        </div>
      )}

      <div>
        <div className="flex items-start">
          <p className="bg-gray-500 py-0.5 px-1.5 text-xs font-bold capitalize text-white">
            {result.type}
          </p>
        </div>
        <div className="pt-1 text-lg">{result.title}</div>
        <div className="pt-1 text-sm text-gray-500">{result.description}</div>
      </div>
    </a>
  )
}
export default SearchResult
