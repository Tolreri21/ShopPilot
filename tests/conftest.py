import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

CASE = {
    "product_id": "B001",
    "title": "iPhone 12 case",
    "brand": "Spigen",
    "bullets": "Slim fit",
    "price": 12.9,
    "in_stock": False,
}


class FakeSearcher:
    def search(self, query, max_price, in_stock, limit):
        return [CASE] if query == "case" else []

    def get_product(self, product_id):
        return CASE if product_id == "B001" else None

    def faq(self, question):
        return [{"question": "Can I return an item?", "answer": "Within 30 days."}]


@pytest.fixture
def searcher():
    return FakeSearcher()


class FakeLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def make_llm():
    return lambda *answers: FakeLLM(messages=iter(answers))
