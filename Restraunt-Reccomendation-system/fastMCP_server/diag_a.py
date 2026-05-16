import asyncio
import time
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_mcp_client():
    server_params = StdioServerParameters(
        command="C:/Users/mjain/Downloads/personal/GenAI-Projects/.venv/Scripts/python.exe",
        args=['server.py'],
        env=os.environ.copy()
    )
    
    try:
        start = time.perf_counter()
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("recommend_by_vibe", arguments={"vibe": "moody"})
                end = time.perf_counter()
                elapsed = end - start
                print(f"MCP_ELAPSED: {elapsed:.4f}")
                print(f"MCP_BUFFER: {str(result)[:250]}")
    except Exception as e:
        print(f"MCP_ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(run_mcp_client())
