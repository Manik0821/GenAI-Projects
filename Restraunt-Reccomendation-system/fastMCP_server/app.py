"""Latency-optimized Gradio host for MCP restaurant recommendations.

This app prioritizes a specialized fast path for common user intents:
recommendations by vibe/preference, restaurant detail lookups, and reviews.
Only ambiguous requests fall back to a short ReAct tool-calling loop.
"""

# SIBLING IMPORT FIX: Dynamically add the current directory to Python path
import sys
from pathlib import Path

current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Libraries to create our MCP host application
import os
import json
import gradio as gr
import httpx
# (Path is already imported above)
from fastmcp.client import Client, PythonStdioTransport
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Sibling local imports now resolve flawlessly
from prompts import SYSTEM_PROMPT, SPECIALIZED_AGENT_PROMPTS
from routing import _heuristic_route, _thinking_message_for_action
from formatting import extract_tool_result_text, format_specialized_result

# ... Rest of your app.py code remains exactly the same ...


def _load_env() -> None:
    """Load .env from project-local and workspace-root locations."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / ".env",          # Restraunt-Reccomendation-system/.env
        script_dir.parent.parent / ".env",   # GenAI-Projects/.env
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)


_load_env()


def _get_nvidia_api_keys() -> list[str]:
    """Return API keys in priority order, de-duplicated."""
    candidates = [
        os.getenv("NVIDIA_API_KEY", ""),
        os.getenv("NVIDIA_API_KEY1", ""),
        os.getenv("NVIDIA_API_KEY2", ""),
        os.getenv("NVIDIA_API_KEY3", ""),
    ]
    keys = []
    for key in candidates:
        key = key.strip()
        if key and key not in keys:
            keys.append(key)
    return keys


_MODEL_BY_KEY: dict[str, ChatOpenAI] = {}
_HTTP_BY_KEY: dict[str, httpx.Client] = {}
_ASYNC_HTTP_BY_KEY: dict[str, httpx.AsyncClient] = {}


def make_model(api_key: str, temperature: float = 0.3) -> ChatOpenAI:
    """Create or reuse a ChatOpenAI client bound to a specific API key."""
    if api_key not in _MODEL_BY_KEY:
        _HTTP_BY_KEY[api_key] = httpx.Client(verify=False, timeout=30.0)
        _ASYNC_HTTP_BY_KEY[api_key] = httpx.AsyncClient(verify=False, timeout=30.0)
        _MODEL_BY_KEY[api_key] = ChatOpenAI(
            model="meta/llama-3.1-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            temperature=temperature,
            max_retries=0,
            http_client=_HTTP_BY_KEY[api_key],
            http_async_client=_ASYNC_HTTP_BY_KEY[api_key],
            model_kwargs={"parallel_tool_calls": False},
        )
    return _MODEL_BY_KEY[api_key]


def _looks_like_auth_or_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    signals = ["401", "403", "429", "unauthorized", "forbidden", "rate", "quota", "api key"]
    return any(s in text for s in signals)


async def _invoke_with_key_failover(messages, tools=None, temperature: float = 0.3):
    """Invoke LLM with ordered API-key failover for auth/quota/rate failures."""
    keys = _get_nvidia_api_keys()
    if not keys:
        raise RuntimeError("No NVIDIA API key found. Set NVIDIA_API_KEY (or NVIDIA_API_KEY1/2/3) in .env.")

    last_exc = None
    for idx, key in enumerate(keys):
        try:
            model = make_model(key, temperature=temperature)
            if tools:
                model = model.bind_tools(tools)
            return await model.ainvoke(messages)
        except Exception as exc:
            last_exc = exc
            # Always try backup keys on auth/quota errors. For first key, also allow a retry path.
            if idx < len(keys) - 1 and (_looks_like_auth_or_quota_error(exc) or idx == 0):
                continue
            break
    raise RuntimeError(f"LLM call failed after key failover attempts: {last_exc}")

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")


def _build_openai_tools(mcp_tools):
    """Convert MCP tool schemas into OpenAI-compatible function specs."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def _route_with_specialized_agent(user_message: str) -> tuple[str, str]:
    """Use deterministic heuristic routing first, then LLM intent routing if needed."""
    base_action, base_arg = _heuristic_route(user_message)
    if base_action != "fallback_react":
        return base_action, base_arg

    messages = [
        SystemMessage(content=SPECIALIZED_AGENT_PROMPTS["intent_router"]),
        HumanMessage(content=user_message),
    ]
    try:
        routed = await _invoke_with_key_failover(messages, temperature=0.0)
        payload = str(routed.content).strip()
        data = json.loads(payload)
        action = data.get("action", "fallback_react")
        argument = data.get("argument", user_message)
        if action in {"recommend_by_vibe", "get_restaurant_info", "get_review"}:
            return action, argument
    except Exception:
        pass
    return "fallback_react", user_message

async def _specialized_fast_path(client: Client, user_message: str, history: list) -> tuple[str, str]:
    """Execute one specialized tool call and format the result. Returns (response, action)."""
    action, argument = await _route_with_specialized_agent(user_message)
    if action == "fallback_react":
        return "", "fallback_react"

    if action == "recommend_by_vibe":
        tool_result = await client.call_tool("recommend_by_vibe", {"vibe": argument})
    elif action == "get_review":
        tool_result = await client.call_tool("get_review", {"restaurant_name": argument})
    else:
        tool_result = await client.call_tool("get_restaurant_info", {"restaurant_name": argument})

    tool_output = extract_tool_result_text(tool_result)
    result = format_specialized_result(action, tool_output)
    return result, action

# MCP Host — ReAct Agent Loop
async def chat_with_agent(user_message: str, history: list) -> tuple[str, str]:
    """Connect to the MCP server, discover tools, and run a ReAct loop.
    Returns (response_text, action_taken). The LLM decides which tools to call, calls them via the MCP server,
    and repeats until it produces a final text response."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        fast_response, action = await _specialized_fast_path(client, user_message, history)
        if fast_response:
            return fast_response, action

        # Discover available tools from the MCP server
        mcp_tools = await client.list_tools()

        # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
        openai_tools = _build_openai_tools(mcp_tools)

        # Build the message list from chat history and the new user message
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # Fallback ReAct loop with tighter turn cap to reduce latency.
        for _ in range(3):
            response = await _invoke_with_key_failover(messages, tools=openai_tools, temperature=0.2)
            messages.append(response)

            # No tool calls means the LLM is done — return the final response
            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    final_text = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                else:
                    final_text = str(raw)
                return final_text, "fallback_react"

            # Execute each tool call via the MCP server and feed results back
            for tool_call in response.tool_calls[:1]:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = extract_tool_result_text(result)
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return "I wasn't able to complete that request. Please try again.", "fallback_react"

# Gradio Event Handler
async def handle_chat(user_message, history):
    """Stream dynamic thinking messages and then final assistant output."""
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    # Start with thinking placeholder; will be updated with action-specific message
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "[*] Processing your request..."},
    ]
    yield history

    # Detect action and extract query argument for context-aware messages
    base_action, extracted_query = _heuristic_route(user_message)
    thinking_msg = _thinking_message_for_action(base_action, extracted_query)
    history[-1] = {"role": "assistant", "content": thinking_msg}
    yield history

    # Execute the query and show final response
    response_text, action_taken = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history

# Gradio Interface
with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.Markdown("# Connoisseur Companion\nYour AI guide to California's restaurant scene. Ask me about restaurants by name, cuisine, or vibe!")

    chatbot = gr.Chatbot(height=500, type="messages", allow_tags=False)
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants", size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot])
    btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot], [chatbot])
    btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot], [chatbot])

# Launch the App
if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
    )
