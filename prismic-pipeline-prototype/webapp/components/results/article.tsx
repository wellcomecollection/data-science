import { formatDateTime } from ".";

export type ArticleResultType = {
  id: string;
  type: "article";
  body: string[];
  contributors?: string[];
  promo_caption: string;
  promo_image: string;
  published: string;
  standfirst: string[];
  title: string;
  partOf: {
    id: string;
    title: string;
    contributors: string[];
  }[];
};

export const ArticleResult = ({
  contributors = [],
  id,
  promo_caption,
  promo_image,
  published,
  standfirst,
  title,
  partOf,
}: ArticleResultType) => {
  return (
    <div className="flex flex-col gap-y-3 gap-x-2 md:flex-row">
      <div className="flex-shrink-0">
        <a
          className="no-underline"
          href={`https://wellcomecollection.org/articles/${id}`}
        >
          <img
            src={promo_image}
            alt={promo_caption}
            className="h-auto w-full rounded object-cover md:h-32 md:w-32"
          />
        </a>
      </div>
      <div className="space-y-1">
        <a
          className="no-underline"
          href={`https://wellcomecollection.org/articles/${id}`}
        >
          <h3 className="pt-0">{title}</h3>

          <p className="text-sm text-dark-gray dark:text-light-gray">
            {formatDateTime(published)}
            {contributors.length > 0 && " - " + contributors.join(", ")}
          </p>
          <p className="text-sm text-dark-gray dark:text-light-gray">
            {standfirst}
          </p>
        </a>
        {partOf.length > 0 &&
          partOf.map((series) => (
            <div className="pt-1 text-sm" key={series.id}>
              Part of{" "}
              <a
                href={`https://wellcomecollection.org/series/${series.id}`}
                className="font-bold underline underline-offset-[3px]"
              >
                {series.title}
              </a>{" "}
              by {series.contributors.join(", ")}
            </div>
          ))}
      </div>
    </div>
  );
};
