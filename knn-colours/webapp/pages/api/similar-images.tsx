import { Client, estypes } from "@elastic/elasticsearch";
import { NextApiRequest, NextApiResponse } from "next";

import { ResultType } from "..";

const index = `images-color-knn-${process.env.INDEX_DATE}`;

const client = new Client({
  node: process.env.ES_PROTOTYPE_HOST as string,
  auth: {
    username: process.env.ES_PROTOTYPE_USERNAME as string,
    password: process.env.ES_PROTOTYPE_PASSWORD as string,
  },
});

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    return res.status(500).json({ error: "Method not allowed" });
  }
  try {
    const queryImage = req.body as ResultType;
    const knnQueries = Object.values(queryImage.colors)
      .map((color) => [
        {
          field: "colors.a.rgb",
          query_vector: color.rgb,
          k: 1000,
          num_candidates: 10000,
        },
        {
          field: "colors.b.rgb",
          query_vector: color.rgb,
          k: 1000,
          num_candidates: 10000,
        },
        {
          field: "colors.c.rgb",
          query_vector: color.rgb,
          k: 1000,
          num_candidates: 10000,
        },
        {
          field: "colors.d.rgb",
          query_vector: color.rgb,
          k: 1000,
          num_candidates: 10000,
        },
        {
          field: "colors.e.rgb",
          query_vector: color.rgb,
          k: 1000,
          num_candidates: 10000,
        },
      ])
      .flat();
    console.log("knnQueries", knnQueries);
    const similarityResponse: estypes.SearchResponse<Document> =
      await client.search({
        index,
        body: {
          // @ts-ignore
          knn: knnQueries,
        },
        size: 7,
      });

    //skip the first result, because it will always be the query image itself
    const similarImages = similarityResponse.hits.hits.slice(1).map((hit) => {
      return {
        id: hit._id,
        ...hit._source,
      };
    });
    res.status(200).json({ similarImages });
  } catch (error) {
    console.error("Failed to fetch similar images:", error);
    res.status(500).json({ message: "Internal Server Error" });
  }
}
