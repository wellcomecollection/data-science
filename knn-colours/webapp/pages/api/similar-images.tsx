import { Client, estypes } from "@elastic/elasticsearch";
import { NextApiRequest, NextApiResponse } from "next";

import { ResultType } from "..";

const index = `images-color-embedding-${process.env.INDEX_DATE}`;

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
    const similarityResponse: estypes.SearchResponse<Document> =
      await client.search({
        index,
        body: {
          knn: {
            field: "embedding",
            // @ts-ignore
            query_vector: queryImage.embedding,
            k: 1000,
            num_candidates: 10000,
          },
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
