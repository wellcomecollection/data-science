import { formatDateTime } from ".";

export type ExhibitionResultType = {
  type: "exhibition";
  id: string;
  title: string;
  body: string[];
  published: string;
  starts: string;
  ends: string;
  promo_image: string;
  promo_caption: string;
  contributors?: string[];
};

export const ExhibitionResult = ({
  id,
  title,
  body,
  published,
  starts,
  ends,
  promo_image,
  promo_caption,
  contributors = [],
}: ExhibitionResultType) => {
  return (
    <a
      className="no-underline"
      href={`https://wellcomecollection.org/exhibitions/${id}`}
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
          <h2 className="pt-0">{title}</h2>
          <div className="text-sm text-dark-gray dark:text-light-gray">
            <p className="pt-0.5">
              {formatDateTime(starts)} - {formatDateTime(ends)}
            </p>
            <p className="pt-1">{promo_caption}</p>
          </div>
        </div>
      </div>
    </a>
  );
};
