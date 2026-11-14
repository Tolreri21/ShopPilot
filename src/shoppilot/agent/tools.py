from langchain.tools import tool

from shoppilot.shop.cart import add_item, cart_contents


def product_line(product: dict) -> str:
    stock = "in stock" if product["in_stock"] else "out of stock"
    return f"{product['product_id']} | {product['title']} | ${product['price']:.2f} | {stock}"


def make_tools(searcher, cart_id: int) -> list:
    @tool(
        description="Search the shop catalog. Returns up to 5 products: id | title | price | stock."
    )
    def search_products(query: str, max_price: float | None = None, in_stock: bool = False) -> str:
        products = searcher.search(query, max_price, in_stock, 5)
        if not products:
            return "No products found."
        return "\n".join(product_line(product) for product in products)

    @tool(description="Get the brand, price, stock and description of one product by its id.")
    def product_details(product_id: str) -> str:
        product = searcher.get_product(product_id)
        if product is None:
            return "Product not found."
        bullets = (product["bullets"] or "")[:1000]
        return f"{product_line(product)}\nBrand: {product['brand']}\n{bullets}"

    @tool(description="Add a product to the customer's cart.")
    def add_to_cart(product_id: str, quantity: int = 1) -> str:
        try:
            add_item(cart_id, product_id, quantity)
        except ValueError as error:
            return f"Could not add: {error}."
        return f"Added {quantity} x {product_id} to the cart."

    @tool(description="Show the items in the customer's cart and the total.")
    def view_cart() -> str:
        contents = cart_contents(cart_id)
        if not contents["items"]:
            return "The cart is empty."
        lines = [
            f"{item['quantity']} x {item['product_id']} | {item['title']} | ${item['price']:.2f}"
            for item in contents["items"]
        ]
        return "\n".join(lines + [f"Total: ${contents['total']:.2f}"])

    @tool(description="Answer questions about shipping, returns, payment and other shop policies.")
    def shop_faq(question: str) -> str:
        return "\n\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in searcher.faq(question))

    return [search_products, product_details, add_to_cart, view_cart, shop_faq]
