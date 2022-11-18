import '../styles/app.css'

import type { AppProps } from 'next/app'

function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />
}

export default App
