import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage

from tools import search_sarees


MODEL_NAME = "llama-3.3-70b-versatile"


def run_agent(image_url: str):

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0
    )

    tools = [
        search_sarees
    ]

    tool_map = {
        "search_sarees": search_sarees
    }

    llm_with_tools = llm.bind_tools(
        tools
    )

    user_message = (
        "Find sarees visually similar to this image: "
        f"{image_url}"
    )

    messages = [
        HumanMessage(
            content=user_message
        )
    ]

    # ==========================================
    # First LLM call
    # ==========================================

    response = llm_with_tools.invoke(
        messages
    )

    messages.append(response)

    tool_result = None

    # ==========================================
    # Execute tool calls
    # ==========================================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        tool = tool_map.get(
            tool_name
        )

        if tool is None:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        tool_result = tool.invoke(
            tool_args
        )

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
        )

    # ==========================================
    # Second LLM call
    # ==========================================

    final_response = llm_with_tools.invoke(
        messages
    )

    return {
        "message": final_response.content,
        "tool_result": tool_result
    }