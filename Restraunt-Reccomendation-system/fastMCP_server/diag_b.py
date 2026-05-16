import asyncio
import time
import sys
import os

async def run_agent_bench():
    try:
        import app
        start = time.perf_counter()
        # Mocking or calling the real app.chat_with_agent
        # Using getattr to be safe if it's not present
        if hasattr(app, "chat_with_agent"):
            response = await app.chat_with_agent("Find me a moody restaurant in LA", [])
            end = time.perf_counter()
            elapsed = end - start
            print(f"AGENT_ELAPSED: {elapsed:.4f}")
            print(f"AGENT_BUFFER: {str(response)[:300]}")
        else:
            print("AGENT_ERROR: app.chat_with_agent not found")
    except Exception as e:
        print(f"AGENT_ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(run_agent_bench())
