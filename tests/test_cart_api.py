import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from shoppilot.api.main import app, get_llm, get_searcher
from shoppilot.shop import cart

client = TestClient(app)


@pytest.fixture
def fake_cart(monkeypatch):
    def cart_contents(cart_id):
        if cart_id != 7:
            raise ValueError("cart not found")
        return {"cart_id": 7, "items": [], "total": 0.0}

    def checkout(cart_id):
        raise ValueError("cart is empty")

    def remove_item(cart_id, product_id):
        cart_contents(cart_id)
        if product_id != "B001":
            raise ValueError("product not found in cart")

    monkeypatch.setattr(cart, "cart_contents", cart_contents)
    monkeypatch.setattr(cart, "checkout", checkout)
    monkeypatch.setattr(cart, "remove_item", remove_item)


def test_get_cart(fake_cart):
    assert client.get("/carts/7").json() == {"cart_id": 7, "items": [], "total": 0.0}
    assert client.get("/carts/8").status_code == 404


def test_checkout_empty_cart_is_400(fake_cart):
    response = client.post("/carts/7/checkout")
    assert response.status_code == 400
    assert response.json()["detail"] == "cart is empty"


def test_remove_from_cart(fake_cart):
    response = client.delete("/carts/7/items/B001")
    assert response.json() == {"cart_id": 7, "items": [], "total": 0.0}
    assert client.delete("/carts/7/items/B002").status_code == 404
    assert client.delete("/carts/8/items/B001").status_code == 404


@pytest.fixture
def fake_agent(searcher, make_llm):
    app.dependency_overrides[get_searcher] = lambda: searcher
    app.dependency_overrides[get_llm] = lambda: make_llm(AIMessage(content="Hi! What do you need?"))
    yield
    app.dependency_overrides.clear()


def test_chat(fake_cart, fake_agent):
    response = client.post("/chat", json={"cart_id": 7, "message": "hello"})
    unknown = client.post("/chat", json={"cart_id": 8, "message": "hello"})
    assert response.json() == {"answer": "Hi! What do you need?"}
    assert unknown.status_code == 404


def test_chat_needs_message(fake_cart, fake_agent):
    assert client.post("/chat", json={"cart_id": 7, "message": ""}).status_code == 422
