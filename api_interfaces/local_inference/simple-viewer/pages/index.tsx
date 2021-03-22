import React, { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { getIndices } from "../services/elastic";
import { GetServerSideProps } from "next";
import { SimilarResponse } from "./api/images/similar";
import usePersistedState from "../src/usePersistedState";
import RadioInput from "../src/RadioInput";
import {
  loadModule,
  ModuleName,
  moduleNames,
  SimilarityModule,
} from "../modules";

const Content = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 10px 25px;
`;

const ImageContainer = styled.div`
  margin: 0 auto;
  width: 300px;
  height: 300px;
  background-color: rgb(235, 235, 235);
`;

const ImagesContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 10px;
`;

const Image = styled.img`
  max-width: 300px;
  width: 100%;
  height: 100%;
  margin: 5px;
`;

const FormContainer = styled.div`
  margin-top: 80px;
  display: flex;
  justify-content: space-around;
`;

const Label = styled.label`
  margin-left: 15px;
`;

const ButtonWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const Button = styled.button`
  padding: 5px 30px;
  font-size: 1.3em;
`;

type RequestType = "random" | "specified" | "by-value";

type Props = {
  indices: string[];
};

const Index: React.FC<Props> = ({ indices }) => {
  const [rootImage, setRootImage] = useState<string | undefined>();
  const [similarImages, setSimilarImages] = useState<string[]>([]);
  const [requestType, setRequestType] = useState<RequestType>("random");
  const [input, setInput] = useState<string>("");
  const [index, setIndex] = usePersistedState<string>("index", indices[0]);
  const [field, setField] = usePersistedState<string>("field", "");
  const [nSimilar, setNSimilar] = usePersistedState<number>("number", 3);
  const [similarityModule, setSimilarityModule] = usePersistedState<
    ModuleName | undefined
  >("similarityModule", undefined);
  const inputToField = useRef<SimilarityModule | undefined>();

  useEffect(() => {
    const doLoadEffect = async () => {
      if (similarityModule) {
        inputToField.current = await loadModule(similarityModule);
      }
    };
    doLoadEffect();
  }, [similarityModule]);

  const setRequestTypeAndClearInput = (value: RequestType) => {
    setInput("");
    setRequestType(value);
  };

  const doRequest = async () => {
    let paramsObject: Record<string, string> = {
      index,
      field,
      n: nSimilar.toString(),
    };
    if (requestType === "specified") {
      paramsObject["id"] = input;
    } else if (requestType === "by-value") {
      const computedFieldValue = inputToField.current?.(input);
      if (computedFieldValue) {
        paramsObject["value"] = JSON.stringify(computedFieldValue);
      }
    }
    const params = new URLSearchParams(paramsObject);
    const result = await fetch("/api/images/similar?" + params.toString());
    if (result.status === 200) {
      const data = (await result.json()) as SimilarResponse;
      setRootImage(data.root?.file);
      setSimilarImages(data.similar.map((s) => s.file));
    }
  };

  return (
    <Content>
      <ImageContainer>{rootImage && <Image src={rootImage} />}</ImageContainer>
      <ImagesContainer>
        {similarImages.map((similarImage) => (
          <Image key={similarImage} src={similarImage} />
        ))}
      </ImagesContainer>
      <FormContainer>
        <div>
          <p>
            <RadioInput<RequestType>
              id="random"
              value={requestType}
              group="request-type"
              setValue={setRequestTypeAndClearInput}
            />
            <Label htmlFor="random">Random image</Label>
          </p>
          <p>
            <RadioInput<RequestType>
              id="specified"
              value={requestType}
              group="request-type"
              setValue={setRequestTypeAndClearInput}
            />
            <Label htmlFor="specified">Specified image</Label>
            {requestType === "specified" && (
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.currentTarget.value)}
              />
            )}
          </p>
          <p>
            <RadioInput<RequestType>
              id="by-value"
              value={requestType}
              group="request-type"
              setValue={setRequestTypeAndClearInput}
            />
            <Label htmlFor="by-value">Calculated field value</Label>
            {requestType === "by-value" && (
              <>
                <select
                  id="module"
                  value={similarityModule || ""}
                  onChange={(e) =>
                    setSimilarityModule(e.currentTarget.value as ModuleName)
                  }
                >
                  <option value="">-</option>
                  {moduleNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.currentTarget.value)}
                />
              </>
            )}
          </p>
        </div>
        <div>
          <p>
            <Label htmlFor="index">Index: </Label>
            <select
              id="index"
              value={index}
              onChange={(e) => setIndex(e.currentTarget.value)}
            >
              {indices.map((i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </p>
          <p>
            <Label htmlFor="field">Field: </Label>
            <input
              type="text"
              value={field}
              onChange={(e) => setField(e.currentTarget.value)}
            />
          </p>
          <p>
            <Label htmlFor="nSimilar">No. similar images: </Label>
            <input
              type="number"
              max="10"
              min="1"
              value={nSimilar}
              onChange={(e) => setNSimilar(parseInt(e.currentTarget.value, 10))}
            />
          </p>
        </div>
        <ButtonWrapper>
          <Button onClick={doRequest}>Go</Button>
        </ButtonWrapper>
      </FormContainer>
    </Content>
  );
};

export const getServerSideProps: GetServerSideProps<Props> = async () => {
  const indices = await getIndices();
  return { props: { indices } };
};

export default Index;
