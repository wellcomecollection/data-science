import { NextPageContext, NextPage } from "next";
import fetch from "isomorphic-unfetch";
import styled from "styled-components";
import Link from "next/link";

type APIImage = { id: string; uri: string };

type APIResponse = {
  original: APIImage;
  neighbours: APIImage[];
};

type Props = APIResponse;

const Flex = styled.div`
  display: flex;
`;

const Palette: NextPage<Props> = ({ neighbours, original }) => {
  return (
    <>
      <h2>Palette</h2>
      <img src={original.uri} alt={""} style={{ width: "100%" }} />
      <Flex>
        {neighbours.map(neighbour => {
          return (
            <Link
              href={{
                pathname: "/palette",
                query: {
                  id: neighbour.id
                }
              }}
            >
              <a>
                <img src={neighbour.uri} alt={""} style={{ width: "100%" }} />
              </a>
            </Link>
          );
        })}
      </Flex>
    </>
  );
};

Palette.getInitialProps = async (ctx: NextPageContext) => {
  const { id } = ctx.query;
  const uri = `https://labs.wellcomecollection.org/palette-api/by_image_id?image_id=${
    id ? id : ""
  }`;

  const resp = await fetch(uri);
  const data = await resp.json();

  return {
    ...data
  };
};

export default Palette;
