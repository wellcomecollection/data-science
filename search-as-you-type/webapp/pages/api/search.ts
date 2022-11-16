import { NextApiRequest, NextApiResponse } from 'next'
import { getClient, search } from '../../services/search'

export default async function searchEndpoint(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method === 'GET') {
    try {
      const { query: searchTerms, n } = req.query
      const nResults = n ? parseInt(n as string) : 10
      const client = getClient()
      const response = await search(client, searchTerms as string, nResults)
      res.status(200).json(response)
    } catch {
      res.status(500).json({ error: 'Unable to query' })
    }
  } else {
    res.status(405).json({ error: 'Method not allowed' })
  }
}
