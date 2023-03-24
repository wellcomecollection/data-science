import { GetServerSideProps, NextPage } from "next";
import { Result, ResultType } from "../components/result";

import Head from "next/head";
import { X as XIcon } from "react-feather";
import { useState } from "react";

type Props = {
  queryParams: { searchTerms?: string; n?: number };
  results?: { [key: string]: ResultType };
};

const Search: NextPage<Props> = (props) => {
  const [searchTerms, setSearchTerms] = useState(props.queryParams.searchTerms);
  const [results, setResults] = useState(props.results);

  return (
    <div className="">
      <Head>
        <link
          rel="icon"
          href={`data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🤔</text></svg>`}
        />
        <title>Semantic stories search</title>
        <meta name="description" content="Very clever semantic search" />
      </Head>
      <main className="min-h-screen">
        <form
          className="border-gray-500 flex w-full divide-x-2 border-2 "
          autoComplete="off"
        >
          <div className="relative mx-auto flex w-full items-center">
            <input
              className="w-full bg-transparent py-2 pl-4 pr-8 text-lg placeholder:text-gray focus:outline-none"
              type="text"
              name="query"
              value={searchTerms}
              autoComplete="off"
              placeholder="What are you looking for?"
              onChange={(e) => setSearchTerms(e.target.value)}
              required
            ></input>
            <button
              type="reset"
              className="absolute right-0 mr-2"
              onClick={() => {
                setSearchTerms("");
                setResults({});
              }}
            >
              <XIcon className="w-6" />
            </button>
          </div>
          <button
            type="submit"
            className="py-2 px-3 text-lg"
            onClick={() => {
              if (searchTerms) {
                window.location.href = `/?query=${searchTerms}`;
              }
            }}
          >
            Search
          </button>
        </form>

        <ul className="flex flex-col gap-y-6 pt-4">
          {results &&
            Object.entries(results).map(([key, result]) => (
              <li key={key}>
                <Result {...result} />
              </li>
            ))}
        </ul>
      </main>
    </div>
  );
};

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  if (!query.query) {
    return {
      props: {
        queryParams: { searchTerms: "" },
        results: null,
      },
    };
  } else {
    const searchResponse = await fetch(
      `http://api:5000/nearest?query=${query.query}&n=${
        query.n ? query.n : 10
      }`,
      {
        method: "GET",
        mode: "no-cors",
      }
    ).then((res) => res.json());

    return {
      props: {
        queryParams: { searchTerms: query.query },
        results: searchResponse.embeddings,
      },
    };
  }
};

export default Search;
