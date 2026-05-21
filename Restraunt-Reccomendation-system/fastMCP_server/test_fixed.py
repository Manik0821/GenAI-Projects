import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
import os

async def run_test():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join("fastMCP_server", "server.py")],
        env={**os.environ, "PYTHONPATH": "."}
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_restaurant_info", arguments={"restaurant_name": "Iron"})
                print("\n--- START SCREENSHOT ---")
                print(result.content[0].text)
                print("--- END SCREENSHOT ---\n")
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
