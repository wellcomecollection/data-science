#!/usr/bin/env python3

import aiohttp
import asyncio
import click
import json
import math
import os
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from palette_adapter import PaletteAdapter

ADAPTERS = {
    "palette": PaletteAdapter
}

# Batching required otherwise asyncio times out
BATCH_SIZE = 1000
MAX_CONNECTIONS = 32


@click.command()
@click.option("-h", "--inferrer-host", type=str, required=True)
@click.option("-a", "--adapter-name", type=click.Choice(list(ADAPTERS.keys())), required=True)
@click.option("-i", "--input-dir", type=str, required=True)
@click.option("-o", "--output-file", type=str, required=True)
@click.option("-n", type=int, default=0)
def run_inference(inferrer_host, adapter_name, input_dir, output_file, n):
    adapter = ADAPTERS[adapter_name](inferrer_host)
    image_urls = [f"file://{os.path.abspath(file.path)}" for file in os.scandir(input_dir)]
    if n != 0:
        image_urls = image_urls[0:n]
    results = []

    async def run():
        connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
        session = aiohttp.ClientSession(connector=connector)

        for batch in tqdm(range(math.ceil(len(image_urls) / BATCH_SIZE))):
            url_batch = image_urls[(batch * BATCH_SIZE):((batch + 1) * BATCH_SIZE)]
            responses = [adapter.make_request(session, image_url) for image_url in url_batch]
            for coroutine in async_tqdm.as_completed(responses):
                result = await coroutine
                results.append(result)

        await session.close()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

    with open(output_file, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    run_inference()
