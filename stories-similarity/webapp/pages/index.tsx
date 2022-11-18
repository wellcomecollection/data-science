import { GetServerSideProps, NextPage } from 'next'

import Favicon from '../components/favicon'
import Head from 'next/head'

type Props = {}

const searchEndpoint = (queryId: string) => `/api/similar/${queryId}&n=6`

const Index: NextPage<Props> = ({}) => {
  return (
    <>
      <Head>
        <title>Stories similarity</title>
        <meta
          name="description"
          content="Prototype UI for surfacing similar stories based on inferred text embeddings"
        />
        <Favicon emoji="📝" />
      </Head>
      <div> hello</div>
    </>
  )
}

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  return { props: {} }
}

export default Index
