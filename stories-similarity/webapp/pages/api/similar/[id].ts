import { NextApiRequest, NextApiResponse } from 'next'
import { getClient, getSimilarStories } from '../../../services'

export default async function searchEndpoint(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method === 'GET') {
    try {
      const { id, n, field } = req.query
      const nResults = n ? parseInt(n as string) : 10
      const queryField = field ? (field as string) : 'title_embedding'
      const client = getClient()
      const response = await getSimilarStories(
        client,
        id as string,
        nResults,
        queryField as string
      )
      res.status(200).json(response)
    } catch {
      res.status(500).json({ error: 'Unable to query' })
    }
  } else {
    res.status(405).json({ error: 'Method not allowed' })
  }
}
