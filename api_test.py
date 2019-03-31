import asyncio
import requests

async def main():
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None, 
            requests.get, 
            'http://localhost:8000/tags-all'
        )
        for i in range(40)
    ]
    for response in await asyncio.gather(*futures):
        pass

loop = asyncio.get_event_loop()
loop.run_until_complete(main())