import { Client } from "@elastic/elasticsearch";

const client = new Client({
  node: process.env["ES_HOST"],
  auth:
    process.env["ES_USER"] && process.env["ES_PASS"]
      ? {
          username: process.env["ES_USER"],
          password: process.env["ES_PASS"],
        }
      : undefined,
});

export type DocResult = {
  id: string;
};

export const getRandomDoc = async (index: string): Promise<DocResult> => {
  const { body } = await client.search({
    index,
    _source: "false",
    body: {
      size: 1,
      query: {
        function_score: {
          random_score: {},
        },
      },
    },
  });
  const hit = body.hits.hits[0];
  return {
    id: hit._id,
  };
};

export const getDoc = async (index: string, id: string): Promise<DocResult> => {
  const { body } = await client.get({ index, id, _source: "image" });
  return {
    id: body._id,
  };
};

type GetSimilarParams = {
  index: string;
  field: string;
  id?: string;
  value?: string | string[];
  n?: number;
};

export const getSimilar = async ({
  index,
  field,
  id,
  value,
  n = 1,
}: GetSimilarParams): Promise<DocResult[]> => {
  const { body } = await client.search({
    index,
    _source: "false",
    body: {
      size: n,
      query: {
        more_like_this: {
          fields: [field],
          like: id
            ? {
                _index: index,
                _id: id,
              }
            : value,
          min_term_freq: 1,
          min_doc_freq: 1,
          max_query_terms: 1000,
          minimum_should_match: 1,
        },
      },
    },
  });

  return body.hits.hits.map((hit: any) => ({
    id: hit._id,
  }));
};

export const getIndices = async (): Promise<string[]> => {
  const { body } = await client.cat.indices({ format: "json" });
  const indices = body.map((idx: any) => idx.index);
  return indices.sort();
};
