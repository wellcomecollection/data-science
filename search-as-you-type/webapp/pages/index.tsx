import Head from 'next/head'

export default function Home() {
  const emojiSvg = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">👀</text></svg>`
  return (
    <>
      <Head>
        <title>Search as you type!</title>
        <meta
          name="description"
          content="Prototype UI for searching as you type"
        />
        <link rel="icon" href={emojiSvg} />
      </Head>
      <div>hello world</div>
    </>
  )
}
