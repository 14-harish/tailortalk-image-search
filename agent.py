import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage

from tools import search_sarees


MODEL_NAME = "llama-3.3-70b-versatile"


# ==========================================
# Check API key
# ==========================================

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set."
    )


# ==========================================
# Load model
# ==========================================

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0
)


# ==========================================
# Tools
# ==========================================

tools = [
    search_sarees
]

tool_map = {
    "search_sarees": search_sarees
}


llm_with_tools = llm.bind_tools(
    tools
)


# ==========================================
# User request
# ==========================================

user_message = (
    "Find sarees visually similar to this image: "
    "https://byrappasilk.in/storage/uploads/"
    "bsrKlEUvx7qmaeA5iC1nEQymK9K4CcA3u9t6LC7G.webp"
)


messages = [
    HumanMessage(
        content=user_message
    )
]


# ==========================================
# First LLM call
# ==========================================

print(
    "\n=============================="
)

print("STEP 1: ASKING AGENT")

print(
    "=============================="
)

response = llm_with_tools.invoke(
    messages
)

messages.append(response)


print("\nAgent tool calls:")

print(response.tool_calls)


# ==========================================
# Execute tool calls
# ==========================================

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]

    tool_args = tool_call["args"]

    tool_call_id = tool_call["id"]

    print(
        f"\nExecuting tool: {tool_name}"
    )

    print(
        f"Arguments: {tool_args}"
    )


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


    print(
        "\nTool returned results."
    )


    messages.append(
        ToolMessage(
            content=str(
                tool_result
            ),
            tool_call_id=tool_call_id
        )
    )


# ==========================================
# Second LLM call
# ==========================================

print(
    "\n=============================="
)

print("STEP 2: AGENT INTERPRETS RESULTS")

print(
    "=============================="
)


final_response = llm_with_tools.invoke(
    messages
)


print(
    "\n=============================="
)

print("FINAL RESPONSE")

print(
    "=============================="
)

print(
    final_response.content
)