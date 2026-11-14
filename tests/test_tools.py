import pytest

from shoppilot.agent.tools import make_tools


@pytest.fixture
def tools(searcher):
    return {t.name: t for t in make_tools(searcher, cart_id=1)}


def test_tool_names(tools):
    assert set(tools) == {
        "search_products",
        "product_details",
        "add_to_cart",
        "view_cart",
        "shop_faq",
    }


def test_search_products(tools):
    result = tools["search_products"].invoke({"query": "case"})
    assert result == "B001 | iPhone 12 case | $12.90 | out of stock"
    assert tools["search_products"].invoke({"query": "sofa"}) == "No products found."


def test_product_details(tools):
    assert "Brand: Spigen" in tools["product_details"].invoke({"product_id": "B001"})
    assert tools["product_details"].invoke({"product_id": "X"}) == "Product not found."


def test_shop_faq(tools):
    assert "30 days" in tools["shop_faq"].invoke({"question": "returns?"})
