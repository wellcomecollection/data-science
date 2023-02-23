import {
  ArticleResult,
  ArticleResultType,
  EventResult,
  EventResultType,
  ExhibitionResult,
  ExhibitionResultType,
} from "../components/results";
import { Client, estypes } from "@elastic/elasticsearch";
import { GetServerSideProps, NextPage } from "next";
import { Tab, tabs } from "@/types/tab";

import Head from "next/head";
import { SearchTotalHits } from "@elastic/elasticsearch/lib/api/types";
import { X as XIcon } from "react-feather";
import articlesQuery from "../../data/queries/articles.json";
import eventsQuery from "../../data/queries/events.json";
import exhibitionsQuery from "../../data/queries/exhibitions.json";
import { useState } from "react";

type Result = ArticleResultType | ExhibitionResultType | EventResultType;

type Props = {
  queryParams: { searchTerms: string; tab: Tab };
  results: Result[];
  total: number;
  took: number;
};

const Search: NextPage<Props> = ({ queryParams, results, total, took }) => {
  const [searchTerms, setSearchTerms] = useState(queryParams.searchTerms);
  const [currentTab, setTab] = useState(queryParams.tab);

  const handleTabChange = (tab: Tab) => {
    setTab(tab);
    // if the search terms are empty, redirect to the tab without the query param
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
          href={`data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">✨</text></svg>`}
        />
        <title>Really good search</title>
        <meta
          name="description"
          content="Search for wellcomecollection.org docs which are maintained in Prismic"
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
                onClick={() => handleTabChange(tab)}
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
              onClick={() => setSearchTerms("")}
            >
              <XIcon className="w-6" />
            </button>
          </div>
          <button
            type="submit"
            className="py-2 px-3 text-lg"
            onClick={() => {
              window.location.href = `/${currentTab}?query=${searchTerms}`;
            }}
          >
            Search
          </button>
        </form>

        <p className="pt-2 text-sm">
          {total} result{total === 1 ? "" : "s"} found in {took}ms
        </p>

        <ul className="flex flex-col gap-y-6 pt-4">
          {results.map((result) => (
            <li key={result.id}>
              {(() => {
                switch (result.type) {
                  case "article":
                    return <ArticleResult {...(result as ArticleResultType)} />;
                  case "exhibition":
                    return (
                      <ExhibitionResult {...(result as ExhibitionResultType)} />
                    );
                  case "event":
                    return <EventResult {...(result as EventResultType)} />;
                }
              })()}
            </li>
          ))}
        </ul>
      </main>
    </div>
  );
};

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  const n = 10;
  if (!query.query) {
    return {
      props: {
        queryParams: { searchTerms: "", tab: query.tab as Tab },
        results: [],
      },
    };
  } else {
    const searchTerms = query.query as string;
    const tab = query.tab as Tab;

    const client = new Client({
      node: process.env.ES_PROTOTYPE_HOST as string,
      auth: {
        username: process.env.ES_PROTOTYPE_USERNAME as string,
        password: process.env.ES_PROTOTYPE_PASSWORD as string,
      },
    });

    const blankQuery = {
      articles: articlesQuery,
      exhibitions: exhibitionsQuery,
      events: eventsQuery,
    }[tab];

    const index = `prismic-${tab}-${process.env.INDEX_DATE}`;
    const searchResponse: estypes.SearchResponse<Document> =
      await client.search({
        index,
        body: JSON.parse(
          JSON.stringify(blankQuery).replace(/{{query}}/g, searchTerms)
        ),
        size: n,
      });

    const results = searchResponse.hits.hits.map((hit: any) => ({
      id: hit._id,
      ...hit._source,
    }));
    const total = (searchResponse.hits.total as SearchTotalHits).value;
    const took = searchResponse.took;

    return {
      props: {
        queryParams: { searchTerms, tab },
        results,
        total,
        took,
      },
    };
  }
};

export default Search;
