import { GetServerSideProps, NextPage } from 'next'
import { Search as SearchIcon, X as XIcon } from 'react-feather'
import { getClient, search } from '../services/search'
import { useCallback, useState } from 'react'

import Favicon from '../components/favicon'
import Head from 'next/head'
import { Result } from '../services/search'
import SearchResult from '../components/search-result'

type Props = {
  queryParams: { searchTerms: string }
  results: Result[]
  total: number
  took: number
}

const searchEndpoint = (query: string) => `/api/search?query=${query}&n=6`

const Search: NextPage<Props> = ({ queryParams, results, total, took }) => {
  const [searchTerms, setSearchTerms] = useState(queryParams.searchTerms)
  const [searchResults, setSearchResults] = useState(results)
  const [searchTotal, setSearchTotal] = useState(total)
  const [searchTook, setSearchTook] = useState(0)

  const handleChange = useCallback((e) => {
    const searchTerms = e.target.value
    setSearchTerms(searchTerms)
    const url = new URL(window.location.href)
    url.searchParams.set('query', searchTerms)
    window.history.pushState({}, '', url.toString())

    if (searchTerms.length > 0) {
      fetch(searchEndpoint(searchTerms))
        .then((res) => res.json())
        .then((res) => {
          setSearchResults(res.results)
          setSearchTotal(res.total)
          setSearchTook(res.took)
        })
    } else {
      setSearchResults([])
      setSearchTotal(0)
      setSearchTook(0)
      const url = new URL(window.location.href)
      url.searchParams.delete('query')
      window.history.pushState({}, '', url.toString())
    }
  }, [])

  return (
    <>
      <Head>
        <title>Search as you type!</title>
        <meta
          name="description"
          content="Prototype UI for searching as you type"
        />
        <Favicon emoji="👀" />
      </Head>
      <div className="flex justify-center pt-8">
        <div className="w-full px-8 lg:w-4/5">
          <form className="block w-full" action={'/'} method="GET">
            <div className="mx-auto flex justify-between gap-1">
              <div className="relative flex w-full items-center border-2 border-black">
                <input
                  className="w-full p-2 pr-8 text-lg focus:outline-none"
                  type="text"
                  name="query"
                  value={searchTerms}
                  placeholder="What are you looking for?"
                  onChange={handleChange}
                  required
                ></input>
                <button
                  type="reset"
                  className="absolute right-0 p-2"
                  onClick={() => {
                    handleChange({ target: { value: '' } })
                  }}
                >
                  <XIcon className="w-5" />
                </button>
              </div>
              <button
                className={`${
                  searchTerms ? 'bg-black text-white' : 'bg-gray-300 text-black'
                } px-3 text-center `}
                type="submit"
              >
                <SearchIcon />
              </button>
            </div>
          </form>
          {searchTotal > 0 && (
            <div className="mt-4">
              <p className="text-sm">
                {`found ${searchTotal} result${
                  searchTotal > 1 ? 's' : ''
                } in ${searchTook}ms`}
              </p>
              <ul className="mt-4 divide-y-2 divide-solid divide-gray-200">
                {searchResults.map((result) => (
                  <li key={result.id} className="list-none py-4">
                    <SearchResult result={result} />
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  if (!query.query) {
    return {
      props: {
        queryParams: { searchTerms: '' },
        results: [],
      },
    }
  } else {
    const client = getClient()
    const response = await search(client, query.query as string, 6)
    return {
      props: {
        queryParams: { searchTerms: query.query as string },
        ...response,
      },
    }
  }
}

export default Search
