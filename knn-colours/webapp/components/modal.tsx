import { Client, estypes } from "@elastic/elasticsearch";

import Image from "next/image";
import { Result } from "./result";
import { ResultType } from "@/pages";
import { type } from "os";

type ModalProps = {
  queryImage: ResultType;
  similarResults: ResultType[];
};
export const Modal = (props: ModalProps) => {
  return (
    <div className="flex flex-row">
      <div className="flex flex-col">
        <Result result={props.queryImage} />
      </div>
      <div className="flex flex-col">
        {props.similarResults.map((result, index) => (
          <Result result={result} key={index} />
        ))}
      </div>
    </div>
  );
};
