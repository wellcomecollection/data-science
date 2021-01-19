class PaletteAdapter:
    def __init__(self, host):
        self.host = host

    async def make_request(self, session, image_url):
        request_url = f"{self.host}/palette/?query_url={image_url}"
        async with session.get(request_url) as response:
            try:
                json = await response.json()
                return {
                    "image": image_url,
                    "response": json
                }
            except Exception as e:
                print(f"failed for {image_url} with:")
                print(e)
                return self.make_request(session, image_url)

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
