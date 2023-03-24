import Link from "next/link";

export type ResultType = {
  id: string;
  type: string;
  title: string;
  text: string;
  embedding: number[];
  score: number;
};

export function formatDateString(date: Date) {
  return date.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

const getLink = (result: ResultType) => {
  let path;
  if (result.type === "webcomics") {
    path = "articles";
  } else {
    path = result.type;
  }

  return `https://wellcomecollection.org/${path}/${result.id}`;
};
export const Result = (result: ResultType) => {
  return (
    <Link href={getLink(result)}>
      <div className="my-1 inline-block bg-light-gray px-1 py-0.5 text-xs font-semibold capitalize dark:bg-gray ">
        {result.type}
      </div>
      <h2>{result.title}</h2>
      <p className="mt-1 border-l-4 border-light-gray pl-4 dark:border-gray dark:text-light-gray ">
        {result.text}
      </p>
    </Link>
  );
};
