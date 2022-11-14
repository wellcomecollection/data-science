import { FunctionComponent } from 'react'
import { Result } from '../services/search'

type Props = {
  result: Result
}

const SearchResult: FunctionComponent<Props> = (props) => {
  return (
    <div>
      <div className="flex items-start">
        <p className="bg-gray-400 py-0.5 px-2 text-sm font-bold capitalize text-white">
          {props.result.type}
        </p>
      </div>

      <div className="pt-2">{props.result.text}</div>
    </div>
  )
}
export default SearchResult
