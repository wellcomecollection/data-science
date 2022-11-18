import { FunctionComponent } from 'react'

type Props = {
  emoji: string
}
const Favicon: FunctionComponent<Props> = ({ emoji }) => {
  const emojiSvg = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">${emoji}</text></svg>`
  return <link rel="icon" href={emojiSvg} />
}
export default Favicon
