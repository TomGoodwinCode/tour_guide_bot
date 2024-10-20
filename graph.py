"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Literal, TypedDict, Union, cast

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from configuration import Configuration
import prompts
from state import InputState, State
from tools import TOOLS
from di.container_instance import bot_container
from utils import load_chat_model
from langchain_cerebras import ChatCerebras

from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)

from langgraph.checkpoint.memory import MemorySaver


def get_prompt_header(guide_role: str) -> str:
    match guide_role:
        case "expert":
            return prompts.EXPERT_HEADER
        case "basic":
            return prompts.BASIC_HEADER
        case "normal":
            return prompts.NORMAL_HEADER


# Define the function that calls the model
async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """
    Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.

    """

    redis_cache = bot_container.redis_cache()

    # get the item of interest from redis

    configuration = Configuration.from_runnable_config(config)
    # Fetch POI data with WikiData
    poi_data = redis_cache.get_item_from_item_store(
        configuration.user_id, configuration.item_id
    )
    if not poi_data:
        raise ValueError(f"Item with item_id {configuration.item_id} not found")

    # Create a prompt template. Customize this to change the agent's behavior.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.system_prompt),
            ("placeholder", "{messages}"),
        ]
    )

    model = ChatCerebras(model="llama3.1-70b", temperature=0).bind_tools(TOOLS)
    # Initialize the model with tool binding. Change the model or add more tools here.
    # model = load_chat_model(configuration.model)
    # model = ChatOpenAI(model="gpt-4o", temperature=0)
    # Prepare the input for the model, including the current system time
    # messages are all but last message
    # input is last message

    tour_length_words = str(int(configuration.tour_length) * 120)

    message_value = await prompt.ainvoke(
        {
            "header": get_prompt_header(configuration.guide_role),
            "tour_length": tour_length_words,
            "messages": state.messages,
            "title": poi_data.title,
            "wiki_title": poi_data.wiki_data.title,
            "wiki_extract": poi_data.wiki_data.extract,
            "wiki_description": poi_data.wiki_data.description,
        },
        config,
    )

    logging.info(f"message_value: {message_value}")
    # Get the model's response
    response = cast(AIMessage, await model.ainvoke(message_value, config))

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }
    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# async def summarize_tour(
#     state: State, config: RunnableConfig
# ) -> Dict[str, List[AIMessage]]:
#     """The tour that the model generates is very word heavy.
#     This model should pull out the most interesting facts, people, and places.
#     This is to give the user context when they are taking the tour"""

#     configuration = Configuration.from_runnable_config(config)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", configuration.summarize_tour_prompt),
#             ("placeholder", "{messages}"),
#         ]
#     )

#     # model = ChatCerebras(model="llama3.1-70b", temperature=0).bind_tools(TOOLS)
#     # Initialize the model with tool binding. Change the model or add more tools here.
#     # model = load_chat_model(configuration.model)
#     model = ChatOpenAI(model="gpt-4o", temperature=0)
#     # Prepare the input for the model, including the current system time
#     # messages are all but last message
#     # input is last message

#     summarisation_value = await prompt.ainvoke(
#         {
#             "messages": state.messages,
#         },
#         config,
#     )

#     logging.info(f"summarisation_value: {summarisation_value}")
#     # Get the model's response
#     response = cast(AIMessage, await model.ainvoke(summarisation_value, config))

#     await adispatch_custom_event(
#         "summary_event", {"data": {"chunk": response.content}}, config=config
#     )

#     return {"summary": [response.content]}


def should_model_continue(state: State, config: RunnableConfig) -> bool:
    """Check if the tour length has been reached.
    Count the words in all the AI messages and compare to the tour length in words. If it is less than the tour length, by 50 words, then prompt the model to continue.
    """
    configuration = Configuration.from_runnable_config(config)
    tour_length_words = str(int(configuration.tour_length) * 120)
    current_word_count = sum(
        len(message.content.split())
        for message in state.messages
        if isinstance(message, AIMessage)
    )
    if current_word_count < int(tour_length_words) - 50:
        return True
    return False


async def prompt_to_continue(
    state: State, config: RunnableConfig
) -> Dict[str, List[SystemMessage]]:
    return {"messages": [SystemMessage(content="Continue your tour.")]}


workflow = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
workflow.add_node(call_model)
workflow.add_node(prompt_to_continue)
# workflow.add_node(summarize_tour)
workflow.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
workflow.add_edge("__start__", "call_model")


def route_model(
    state: State, config: RunnableConfig
) -> Literal["prompt_to_continue", "tools", "END"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("check_tour_length" or "tools").
    """

    last_message = state.messages[-1]
    logging.info(last_message)
    if not isinstance(last_message, AIMessage) and not isinstance(
        last_message, SystemMessage
    ):
        raise ValueError(
            f"Expected AIMessage or SystemMessage in output edges, but got {type(last_message).__name__}"
        )

    # If there is no tool call, then we finish
    if last_message.tool_calls:
        return "tools"
    if should_model_continue(state, config):
        return "prompt_to_continue"
    return "END"


# Add a conditional edge to determine the next step after `call_model`
workflow.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model
    route_model,
    {
        "prompt_to_continue": "prompt_to_continue",
        "tools": "tools",
        "END": END,
    },
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
workflow.add_edge("tools", "call_model")
workflow.add_edge("prompt_to_continue", "call_model")


memory = MemorySaver()

# Compile the workflow into an executable graph
# You can customize this by adding interrupt points for state updates
graph = workflow.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
    debug=True,
    checkpointer=memory,
)
graph.name = "Pipecat Bot 2"  # This customizes the name in LangSmith
