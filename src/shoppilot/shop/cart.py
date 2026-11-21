from sqlalchemy import select

from shoppilot.db.models import Cart, CartItem, Order, Product
from shoppilot.db.session import Session


def create_cart() -> int:
    with Session() as session:
        cart = Cart()
        session.add(cart)
        session.commit()
        return cart.id


def check_cart(session, cart_id: int) -> None:
    if session.get(Cart, cart_id) is None:
        raise ValueError("cart not found")
    if session.scalar(select(Order).where(Order.cart_id == cart_id)) is not None:
        raise ValueError("cart is already checked out")


def add_item(cart_id: int, product_id: str, quantity: int = 1) -> None:
    if quantity < 1:
        raise ValueError("quantity must be at least 1")
    with Session() as session:
        check_cart(session, cart_id)
        product = session.get(Product, product_id)
        if product is None:
            raise ValueError("product not found")
        if not product.in_stock:
            raise ValueError("product is out of stock")
        item = session.scalar(
            select(CartItem).where(CartItem.cart_id == cart_id, CartItem.product_id == product_id)
        )
        if item is None:
            session.add(CartItem(cart_id=cart_id, product_id=product_id, quantity=quantity))
        else:
            item.quantity += quantity
        session.commit()


def remove_item(cart_id: int, product_id: str) -> None:
    with Session() as session:
        check_cart(session, cart_id)
        item = session.scalar(
            select(CartItem).where(CartItem.cart_id == cart_id, CartItem.product_id == product_id)
        )
        if item is None:
            raise ValueError("product not found in cart")
        session.delete(item)
        session.commit()


def cart_contents(cart_id: int) -> dict:
    with Session() as session:
        if session.get(Cart, cart_id) is None:
            raise ValueError("cart not found")
        rows = session.execute(
            select(CartItem, Product).join(Product).where(CartItem.cart_id == cart_id)
        ).all()
    items = [
        {
            "product_id": product.product_id,
            "title": product.title,
            "price": product.price,
            "quantity": item.quantity,
        }
        for item, product in rows
    ]
    total = round(sum(item["price"] * item["quantity"] for item in items), 2)
    return {"cart_id": cart_id, "items": items, "total": total}


def checkout(cart_id: int) -> dict:
    contents = cart_contents(cart_id)
    if not contents["items"]:
        raise ValueError("cart is empty")
    with Session() as session:
        check_cart(session, cart_id)
        order = Order(cart_id=cart_id, total=contents["total"])
        session.add(order)
        session.commit()
        return {"order_id": order.id, "total": order.total}
