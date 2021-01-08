import { NextApiRequest, NextApiResponse } from "next";
import { getDoc, getRandomDoc, getSimilar } from "../../../services/elastic";

type ImageResponse = {
  id: string;
  file: string;
};

export type SimilarResponse = {
  root: ImageResponse;
  similar: ImageResponse[];
};

type ErrorResponse = {
  error: string;
};

const singleQ = (q: string | string[] | undefined): q is string =>
  Boolean(q && typeof q === "string");

const idToFile = (id: string): string =>
  `${process.env["IMAGE_HOST"]}/${id}.jpg`;

export default async (
  req: NextApiRequest,
  res: NextApiResponse<SimilarResponse | ErrorResponse>
) => {
  const index = req.query["index"];
  if (!singleQ(index)) {
    return res.status(400).json({ error: "No index specified" });
  }

  const field = req.query["field"];
  if (!singleQ(field)) {
    return res.status(400).json({ error: "No field specified" });
  }

  const root = await ("id" in req.query && typeof req.query.id === "string"
    ? getDoc(index, req.query["id"])
    : getRandomDoc(index));
  const result = await getSimilar({
    index,
    field,
    id: root.id,
    n: singleQ(req.query["n"]) ? parseInt(req.query["n"], 10) : 1,
  });

  const rootResponse = {
    ...root,
    file: idToFile(root.id),
  };
  const resultResponse: ImageResponse[] = result.map(({ id }) => ({
    id,
    file: idToFile(id),
  }));

  res.status(200).json({
    root: rootResponse,
    similar: resultResponse,
  });
};
