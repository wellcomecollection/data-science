import { formatDateTime } from ".";

export type ArticleResultType = {
  type: "article";
  body: string[];
  contributors?: string[];
  id: string;
  promo_caption: string;
  promo_image: string;
  published: string;
  standfirst: string[];
  title: string;
};

export const ArticleResult = ({
  body,
  contributors = [],
  id,
  promo_caption,
  promo_image,
  published,
  standfirst,
  title,
}: ArticleResultType) => {
  return (
    <a
      className="no-underline"
      href={`https://wellcomecollection.org/articles/${id}`}
    >
      <div className="flex flex-col gap-y-3 gap-x-2 md:flex-row ">
        <div className="flex-shrink-0">
          <img
            src={promo_image}
            alt={promo_caption}
            className="h-auto w-full rounded object-cover md:h-32 md:w-32"
          />
        </div>
        <div>
          <h3 className="pt-0">{title}</h3>
          <div className="text-sm text-dark-gray dark:text-light-gray">
            <p className="pt-0.5">
              {formatDateTime(published)}
              {contributors.length > 0 && " - " + contributors.join(", ")}
            </p>
            <p className="pt-1">{standfirst}</p>
          </div>
        </div>
      </div>
    </a>
  );
};
