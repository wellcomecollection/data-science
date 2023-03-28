import Link from "next/link";

export type ResultType = {
  id: string;
  type: string;
  format: string;
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
  if (result.type === "works") {
    return `https://wellcomecollection.org/works/${result.id}`;
  } else if (result.format === "webcomics") {
    return `https://wellcomecollection.org/articles/${result.id}`;
  } else {
    return `https://wellcomecollection.org/${result.format}/${result.id}`;
  }
};

const getLabel = (result: ResultType) => {
  if (result.type === "works") {
    return result.format;
  } else if (result.format === "series") {
    return "series";
  } else {
    return result.format.replace(/s$/, "");
  }
};

export const Result = (result: ResultType) => {
  return (
    <Link href={getLink(result)}>
      <div className="flex items-center gap-x-2 text-xs">
        <div className="my-1 inline-block bg-light-gray px-1 py-0.5 font-semibold capitalize dark:bg-gray ">
          {getLabel(result)}
        </div>
        <div className="text-gray dark:text-light-gray">
          {Math.round(result.score * 100)}% match
        </div>
      </div>
      <h2>{result.title}</h2>
      {result.title != result.text && (
        <p className="mt-1 border-l-4 border-light-gray pl-4 dark:border-gray dark:text-light-gray ">
          {result.text}
        </p>
      )}
    </Link>
  );
};
