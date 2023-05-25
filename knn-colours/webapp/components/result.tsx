import Image from "next/image";
import { ResultType } from "@/pages";

export const Result = ({ result }: { result: ResultType }) => {
  return (
    <div>
      <Image
        src={result.thumbnail_url}
        alt={result.title}
        width={1000}
        height={1000}
      />
    </div>
  );
};
