from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from shoppilot.agent.graph import ask, build_agent
from shoppilot.agent.tools import make_tools


class FakeLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_calls_search_tool(searcher):
    llm = FakeLLM(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_products", "args": {"query": "case"}, "id": "1"}],
                ),
                AIMessage(content="B001 iPhone 12 case costs $12.90."),
            ]
        )
    )
    agent = build_agent(llm, make_tools(searcher, cart_id=1))
    assert ask(agent, 1001, "I need a phone case") == "B001 iPhone 12 case costs $12.90."

    state = agent.get_state({"configurable": {"thread_id": "1001"}})
    tool_message = state.values["messages"][2]
    assert tool_message.content == "B001 | iPhone 12 case | $12.90 | out of stock"
