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
      <div className="flex h-2 flex-row flex-wrap">
        <div
          key={result.colors.a.hex}
          className="w-1/5"
          style={{ backgroundColor: result.colors.a.hex }}
        />
        <div
          key={result.colors.b.hex}
          className="w-1/5"
          style={{ backgroundColor: result.colors.b.hex }}
        />
        <div
          key={result.colors.c.hex}
          className="w-1/5"
          style={{ backgroundColor: result.colors.c.hex }}
        />
        <div
          key={result.colors.d.hex}
          className="w-1/5"
          style={{ backgroundColor: result.colors.d.hex }}
        />
        <div
          key={result.colors.e.hex}
          className="w-1/5"
          style={{ backgroundColor: result.colors.e.hex }}
        />
      </div>
    </div>
  );
};
