import pytest
from fastapi.testclient import TestClient

from shoppilot.api.main import app, get_searcher

PHONE_CASE = {
    "product_id": "B001",
    "title": "iPhone 12 case",
    "brand": "Spigen",
    "bullets": None,
    "price": 12.99,
    "in_stock": True,
    "score": 0.9,
}


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, max_price, in_stock, limit):
        self.calls.append((query, max_price, in_stock, limit))
        return [PHONE_CASE]

    def get_product(self, product_id):
        return PHONE_CASE if product_id == "B001" else None


@pytest.fixture
def fake():
    searcher = FakeSearcher()
    app.dependency_overrides[get_searcher] = lambda: searcher
    yield searcher
    app.dependency_overrides.clear()


client = TestClient(app)


def test_search_passes_filters(fake):
    response = client.get("/search", params={"q": "case", "max_price": 20, "in_stock": True})
    assert response.status_code == 200
    assert response.json()[0]["product_id"] == "B001"
    assert fake.calls == [("case", 20.0, True, 10)]


def test_search_needs_query(fake):
    assert client.get("/search").status_code == 422


def test_limit_is_capped(fake):
    assert client.get("/search", params={"q": "case", "limit": 500}).status_code == 422


def test_get_product(fake):
    assert client.get("/products/B001").json()["title"] == "iPhone 12 case"


def test_unknown_product_is_404(fake):
    assert client.get("/products/nope").status_code == 404
