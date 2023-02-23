export function formatDateTime(dateString: string) {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-GB", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

import { ArticleResult, ArticleResultType } from "./article";
import { EventResult, EventResultType } from "./event";
import { ExhibitionResult, ExhibitionResultType } from "./exhibition";

export {
  ArticleResult,
  EventResult,
  ExhibitionResult,
  ArticleResultType,
  EventResultType,
  ExhibitionResultType,
};
