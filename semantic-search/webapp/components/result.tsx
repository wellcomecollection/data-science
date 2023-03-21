import Link from "next/link";

export type ResultType = {
  id: string;
  text: string;
  embedding: number[];
};

export const Result = ({ id, text, embedding }: ResultType) => {
  return (
    <Link href={`https://wellcomecollection.org/articles/${id}`}>
      <h2>{id}</h2>
      <p>{text}</p>
    </Link>
  );
};
