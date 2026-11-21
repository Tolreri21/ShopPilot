from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from shoppilot.config import settings

SYSTEM_PROMPT = """You are ShopPilot, a shopping assistant for an online shop.
Use search_products to find products and only recommend products returned by the tools.
Always show the product id and price. Never make up products, prices or shop policies.
Use shop_faq for questions about shipping, returns, payment and other policies.
Add products to the cart only when the customer asks for it.
You cannot place orders. When the customer is ready, tell them to press the checkout button.
Answer briefly."""

memory = InMemorySaver()


def load_llm():
    if not settings.gemini_api_key:
        raise RuntimeError("set GEMINI_API_KEY in .env")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model, google_api_key=settings.gemini_api_key, temperature=0
    )


def build_agent(llm, tools):
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT, checkpointer=memory)


def run(agent, cart_id: int, message: str) -> list:
    config = {"configurable": {"thread_id": str(cart_id)}}
    return agent.invoke({"messages": [{"role": "user", "content": message}]}, config)["messages"]


def ask(agent, cart_id: int, message: str) -> str:
    return run(agent, cart_id, message)[-1].text
