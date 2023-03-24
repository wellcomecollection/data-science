import Link from "next/link";

export type ResultType = {
  id: string;
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

export const Result = (props: ResultType) => {
  return (
    <Link href={`https://wellcomecollection.org/articles/${props.id}`}>
      <h2>{props.title}</h2>
      <p>{props.score}</p>
      <p className="mt-1 border-l-4 border-gray pl-4 dark:text-light-gray ">
        {props.text}
      </p>
    </Link>
  );
};
