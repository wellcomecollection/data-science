import { GetServerSideProps, NextPage } from "next";

import Head from "next/head";
import Image from "next/image";
import { X as XIcon } from "react-feather";
import { useState } from "react";

export type ResultType = {
  score: number;
  image_id: string;
  source_id: string;
  title: string;
  thumbnail_url: string;
  embedding?: number[];
};

type Props = {
  queryParams: { searchTerms?: string; n?: number };
  results?: { [key: string]: ResultType };
  took?: number;
  total?: number;
};

const Search: NextPage<Props> = (props) => {
  const [searchTerms, setSearchTerms] = useState(props.queryParams.searchTerms);
  const [results, setResults] = useState(props.results);

  return (
    <div>
      <Head>
        <link
          rel="icon"
          href={`data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">👁️</text></svg>`}
        />
        <title>CLIP search</title>
        <meta
          name="description"
          content="Image search without image captions"
        />
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

        {results && (
          <p className="pt-2 text-sm">
            found {props.total} result{props.total == 1 ? "" : "s"} in{" "}
            {props.took}ms
          </p>
        )}

        <div className="pt-4">
          <ul className="grid grid-flow-dense grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4	">
            {props.results &&
              Object.values(props.results).map((result) => (
                <li key={result.image_id}>
                  <a
                    href={`https://wellcomecollection.org/works/${result.source_id}/images?id=${result.image_id}`}
                  >
                    <Image
                      src={result.thumbnail_url}
                      alt={result.title}
                      width={1000}
                      height={1000}
                    />
                  </a>
                </li>
              ))}
          </ul>
        </div>
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
        total: null,
        took: null,
      },
    };
  } else {
    // encode the search terms
    const searchResponse = await fetch(
      `${process.env.API_URL}/images?query=${query.query}&n=${
        query.n ? query.n : 10
      }`,
      {
        method: "GET",
        mode: "no-cors",
      }
    ).then((res) => res.json());

    return {
      props: {
        tab: query.index,
        queryParams: { searchTerms: query.query },
        results: searchResponse.results,
        total: searchResponse.total,
        took: searchResponse.took,
      },
    };
  }
};

export default Search;
