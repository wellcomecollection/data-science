import { FunctionComponent } from 'react'
import { Result } from '../services/search'

type Props = {
  result: Result
}

const SearchResult: FunctionComponent<Props> = (props) => {
  return (
    <div>
      <div className="flex items-start">
        <p className="bg-gray-500 py-0.5 px-1.5 text-xs font-bold capitalize text-white">
          {props.result.type}
        </p>
      </div>

      <div className="pt-1 font-bold">{props.result.title}</div>
      <div className="pt-1 text-sm text-gray-500">
        {props.result.description}
      </div>
    </div>
  )
}
export default SearchResult
