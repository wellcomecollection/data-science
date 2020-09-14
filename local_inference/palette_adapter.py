class PaletteAdapter:
    def __init__(self, host):
        self.host = host

    async def make_request(self, session, image_url):
        request_url = f"{self.host}/palette/?query_url={image_url}"
        async with session.get(request_url) as response:
            json = await response.json()
            return {
                "image": image_url,
                "response": json
            }

    @staticmethod
    def get_mapping():
        return {
            "image": {"type": "text"},
            "response": {
                "properties": {
                    "palette": {"type": "keyword"}
                }
            }
        }
