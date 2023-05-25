import { Client, estypes } from "@elastic/elasticsearch";
import { GetServerSideProps, NextPage } from "next";
import { useEffect, useState } from "react";

import Head from "next/head";
import Image from "next/image";
import { Result } from "@/components/result";
import { SearchTotalHits } from "@elastic/elasticsearch/lib/api/types";
import { X as XIcon } from "react-feather";
import { hex_to_rgb } from "@/helpers/colours";
import { url } from "inspector";

export type ResultColorType = {
  lab: number[];
  rgb: number[];
  proportion: number;
  hex: string;
};

export type ResultType = {
  id: string;
  source_id: string;
  title: string;
  thumbnail_url: string;
  colors: {
    a: ResultColorType;
    b: ResultColorType;
    c: ResultColorType;
    d: ResultColorType;
    e: ResultColorType;
  };
};

type Props = {
  searchTerms: string;
  color: string;
  results: ResultType[] | null;
  total: number;
  took: number;
};

const index = `images-color-knn-${process.env.INDEX_DATE}`;

const client = new Client({
  node: process.env.ES_PROTOTYPE_HOST as string,
  auth: {
    username: process.env.ES_PROTOTYPE_USERNAME as string,
    password: process.env.ES_PROTOTYPE_PASSWORD as string,
  },
});

const Search: NextPage<Props> = (props) => {
  const [searchTerms, setSearchTerms] = useState(props.searchTerms);
  const [color, setColor] = useState(props.color);
  const [modal, setModal] = useState(false);
  const [queryImage, setQueryImage] = useState<ResultType | null>(null);
  const [similarImages, setSimilarImages] = useState<ResultType[]>([]);

  async function handleImageClick(image: ResultType) {
    setQueryImage(image);
    setModal(true);
    try {
      const response = await fetch("/api/similar-images", {
        method: "POST",
        body: JSON.stringify(image),
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Request failed");
      }

      const { similarImages }: { similarImages: ResultType[] } =
        await response.json();
      setSimilarImages(similarImages);
    } catch (error) {
      console.error("Failed to fetch similar images:", error);
    }
  }

  return (
    <div className="">
      <Head>
        <link
          rel="icon"
          href={`data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🎨</text></svg>`}
        />
        <title>Colours!</title>
        <meta
          name="description"
          content="Colour search and similarity using elasticsearch KNN queries"
        />
      </Head>
      <main className="min-h-screen">
        <form autoComplete="off">
          <div className="border-gray-500 flex w-full divide-x-2 border-2">
            <div className="relative mx-auto flex w-full items-center">
              <input
                className="w-full bg-transparent py-2 pl-4 pr-8 text-lg placeholder:text-gray focus:outline-none"
                type="text"
                name="query"
                value={searchTerms}
                autoComplete="off"
                placeholder="What are you looking for?"
                onChange={(e) => setSearchTerms(e.target.value)}
              />
              <button
                type="reset"
                className="absolute right-0 mr-2"
                onClick={() => setSearchTerms("")}
              >
                <XIcon className="w-6" />
              </button>
            </div>
            <label className="flex items-center text-lg">
              <input
                type="color"
                name="color"
                value={color}
                onChange={(e) => setColor(e.target.value)}
                className="sr-only"
              />
              <span className="inline-block h-full w-12">
                <span
                  className="block h-full w-full"
                  style={{ backgroundColor: color }}
                ></span>
              </span>
            </label>
            <button type="submit" className="px-3 text-lg">
              Search
            </button>
          </div>
        </form>

        {props.results && (
          <p className="pt-2 text-sm">
            <span>
              {props.total} result{props.total === 1 ? "" : "s"} found in{" "}
              {props.took}ms
            </span>
          </p>
        )}

        <div className="pt-4">
          <ul className="columns-3 space-y-6 lg:columns-4">
            {props.results &&
              props.results.map((result) => (
                <li key={result.id} onClick={() => handleImageClick(result)}>
                  <Result result={result} />
                </li>
              ))}
          </ul>
        </div>
        {modal && queryImage && (
          <div
            className={`fixed inset-0 z-50 ${
              modal ? "opacity-100" : "pointer-events-none opacity-0"
            }`}
            onClick={() => {
              setModal(false);
              setQueryImage(null);
              setSimilarImages([]);
            }}
          >
            <div className="absolute inset-0 bg-black opacity-75 "></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="relative h-5/6 w-5/6 rounded-lg bg-light-gray dark:bg-gray">
                <div className="flex space-x-8 p-8">
                  <div className="w-100 lg:w-1/2">
                    <h2 className="text-lg font-medium">{queryImage.title}</h2>
                    <a
                      href={`https://wellcomecollection.org/works/${queryImage.source_id}/images?id=${queryImage?.id}`}
                    >
                      <Image
                        src={queryImage.thumbnail_url}
                        alt={queryImage.title}
                        width={1000}
                        height={1000}
                      />
                    </a>
                  </div>
                  <div className="w-100  lg:float-right lg:w-1/2">
                    <h3 className="text-lg font-medium">Palette</h3>
                    <div className="flex h-12 flex-row flex-wrap space-x-2">
                      <a
                        href={`?color=${encodeURIComponent(
                          queryImage.colors.a.hex
                        )}&query=`}
                        key={queryImage.colors.a.hex}
                        className="w-12"
                        style={{ backgroundColor: queryImage.colors.a.hex }}
                      />

                      <a
                        href={`?color=${encodeURIComponent(
                          queryImage.colors.b.hex
                        )}&query=`}
                        key={queryImage.colors.b.hex}
                        className="w-12"
                        style={{ backgroundColor: queryImage.colors.b.hex }}
                      />
                      <a
                        href={`?color=${encodeURIComponent(
                          queryImage.colors.c.hex
                        )}&query=`}
                        key={queryImage.colors.c.hex}
                        className="w-12"
                        style={{ backgroundColor: queryImage.colors.c.hex }}
                      />
                      <a
                        href={`?color=${encodeURIComponent(
                          queryImage.colors.d.hex
                        )}&query=`}
                        key={queryImage.colors.d.hex}
                        className="w-12"
                        style={{ backgroundColor: queryImage.colors.d.hex }}
                      />
                      <a
                        href={`?color=${encodeURIComponent(
                          queryImage.colors.e.hex
                        )}&query=`}
                        key={queryImage.colors.e.hex}
                        className="w-12"
                        style={{ backgroundColor: queryImage.colors.e.hex }}
                      />
                    </div>
                    <h3 className="text-lg font-medium">
                      Images with similar colours
                    </h3>

                    <div className="columns-3 space-y-6">
                      {similarImages.map((result) => (
                        <div key={result.id}>
                          <a
                            href={`https://wellcomecollection.org/works/${result.source_id}/images?id=${result.id}`}
                          >
                            <Image
                              src={result.thumbnail_url}
                              alt={result.title}
                              width={1000}
                              height={1000}
                            />
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export const getServerSideProps: GetServerSideProps = async ({
  query: queryParams,
}) => {
  const n = 25;
  const searchTerms = queryParams.query as string;
  const color = queryParams.color as string;

  if (!searchTerms && !color) {
    return {
      props: {
        searchTerms: "",
        color: "#facc15",
        results: null,
        total: 0,
        took: 0,
      },
    };
  }
  let elasticsearchQuery: any = {};

  if (searchTerms) {
    elasticsearchQuery.query = {
      match: {
        title: searchTerms,
      },
    };
  }
  if (color) {
    const rgb_vector = hex_to_rgb(color);
    elasticsearchQuery.knn = [
      {
        field: "colors.a.rgb",
        query_vector: rgb_vector,
        k: 100,
        num_candidates: 100,
      },
      {
        field: "colors.b.rgb",
        query_vector: rgb_vector,
        k: 100,
        num_candidates: 100,
      },
      {
        field: "colors.c.rgb",
        query_vector: rgb_vector,
        k: 100,
        num_candidates: 100,
      },
      {
        field: "colors.d.rgb",
        query_vector: rgb_vector,
        k: 100,
        num_candidates: 100,
      },
      {
        field: "colors.e.rgb",
        query_vector: rgb_vector,
        k: 100,
        num_candidates: 100,
      },
    ];
  }

  const searchResponse: estypes.SearchResponse<Document> = await client.search({
    index,
    body: elasticsearchQuery,
    size: n,
  });

  const results = searchResponse.hits.hits.map((hit: any) => {
    return { id: hit._id, ...hit._source };
  });
  const total = (searchResponse.hits.total as SearchTotalHits).value;
  const took = searchResponse.took;

  return {
    props: {
      searchTerms,
      color,
      results,
      total,
      took,
    },
  };
};

export default Search;
