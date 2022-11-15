import { ArrowUpRight, Search as SearchIcon, X as XIcon } from 'react-feather'
import { GetServerSideProps, NextPage } from 'next'
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

const searchEndpoint = (searchTerms: string) =>
  `/api/search?query=${searchTerms}&n=6`

const Search: NextPage<Props> = ({ queryParams, results, total, took }) => {
  const [searchTerms, setSearchTerms] = useState(queryParams.searchTerms)
  const [searchResults, setSearchResults] = useState(results)
  const [searchTotal, setSearchTotal] = useState(total)
  const [searchTook, setSearchTook] = useState(took)

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
      <div className="flex justify-center pt-4 lg:pt-8">
        <div className="w-full px-6 lg:w-4/5">
          <p className="text-sm">
            {searchTotal
              ? `found ${searchTotal} result${
                  searchTotal !== 1 ? 's' : ''
                } in ${searchTook}ms`
              : 'no results'}
          </p>
          <form action={'/'} method="GET" className="pt-2">
            <div className="mx-auto flex justify-between gap-1">
              <div className="relative flex w-full items-center border-2 border-gray-500">
                <input
                  className="w-full py-2 pl-4 pr-8 text-lg focus:outline-none"
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
            </div>
            {searchTotal > 0 && (
              <ul className="divide-y-2 divide-gray-100 overflow-hidden rounded-b-lg border-x-2 border-b-2 border-gray-500">
                <a
                  href={`https://wellcomecollection.org/works?query=${searchTerms}`}
                >
                  <li
                    key={'search'}
                    className="flex list-none items-center p-4 text-gray-500 hover:bg-gray-100"
                  >
                    {`Search for "${searchTerms}"`}{' '}
                    <ArrowUpRight className="ml-2 h-5" />
                  </li>
                </a>
                {searchResults.map((result) => (
                  <li key={result.id} className="list-none ">
                    <SearchResult result={result} />
                  </li>
                ))}
              </ul>
            )}
          </form>
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
        total: 0,
        took: 0,
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
