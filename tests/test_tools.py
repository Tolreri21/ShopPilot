from shoppilot.agent.tools import make_tools

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


def get_tools():
    return {t.name: t for t in make_tools(FakeSearcher(), cart_id=1)}


def test_tool_names():
    assert set(get_tools()) == {
        "search_products",
        "product_details",
        "add_to_cart",
        "view_cart",
        "shop_faq",
    }


def test_search_products():
    tools = get_tools()
    result = tools["search_products"].invoke({"query": "case"})
    assert result == "B001 | iPhone 12 case | $12.90 | out of stock"
    assert tools["search_products"].invoke({"query": "sofa"}) == "No products found."


def test_product_details():
    tools = get_tools()
    assert "Brand: Spigen" in tools["product_details"].invoke({"product_id": "B001"})
    assert tools["product_details"].invoke({"product_id": "X"}) == "Product not found."


def test_shop_faq():
    assert "30 days" in get_tools()["shop_faq"].invoke({"question": "returns?"})
