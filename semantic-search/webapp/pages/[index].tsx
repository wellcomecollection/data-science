import { GetServerSideProps, NextPage } from "next";
import { Result, ResultType } from "../components/result";

import Head from "next/head";
import { X as XIcon } from "react-feather";
import { useState } from "react";

type Tab = "works" | "prismic";
const tabs = ["works", "prismic"];
type Props = {
  queryParams: { searchTerms?: string; n?: number };
  results?: { [key: string]: ResultType };
  took?: number;
  total?: number;
  tab: Tab;
};

const Search: NextPage<Props> = (props) => {
  const [searchTerms, setSearchTerms] = useState(props.queryParams.searchTerms);
  const [results, setResults] = useState(props.results);
  const [currentTab, setTab] = useState(props.tab);

  const handleTabChange = (tab: Tab) => {
    setTab(tab);
    if (searchTerms === "") {
      window.location.href = `/${tab}`;
    } else {
      window.location.href = `/${tab}?query=${searchTerms}`;
    }
  };

  return (
    <div className="">
      <Head>
        <link
          rel="icon"
          href={`data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🤔</text></svg>`}
        />
        <title>Semantic search</title>
        <meta
          name="description"
          content="Very clever search big brain wellcome collection"
        />
      </Head>
      <main className="min-h-screen">
        <div className="flex flex-row">
          <div className="divide-x-2 border-2 border-b-0">
            {tabs.map((tab) => (
              <button
                key={tab}
                className={`${
                  tab === currentTab
                    ? "underline decoration-yellow-400 decoration-8 underline-offset-8"
                    : ""
                } px-4 pt-2 pb-4 text-lg capitalize`}
                onClick={() => handleTabChange(tab as Tab)}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

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

        <ul className="flex flex-col space-y-4 divide-y divide-light-gray py-6 dark:divide-gray">
          {results &&
            Object.entries(results).map(([key, result]) => (
              <li key={key} className="pt-4">
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
        total: null,
        took: null,
      },
    };
  } else {
    // encode the search terms

    const searchResponse = await fetch(
      `http://api:5000/${query.index}?query=${query.query}&n=${
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
