import pytest
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.exc import OperationalError

from alembic import command
from shoppilot.db.models import Cart, CartItem, Order, Product
from shoppilot.db.session import Session, engine
from shoppilot.shop.cart import add_item, cart_contents, checkout, create_cart

TEST_PRODUCTS = ["TEST-CASE", "TEST-SOLD-OUT"]


@pytest.fixture
def cart_id():
    try:
        engine.connect().close()
    except OperationalError:
        pytest.skip("Postgres is not running, start it with `make up`")
    command.upgrade(Config("alembic.ini"), "head")

    with Session() as session:
        for product_id, in_stock in zip(TEST_PRODUCTS, [True, False]):
            session.merge(
                Product(
                    product_id=product_id,
                    title=product_id,
                    price=10.5,
                    price_source="test",
                    in_stock=in_stock,
                    embedding=[0.0] * 384,
                )
            )
        session.commit()

    cart_id = create_cart()
    yield cart_id

    with Session() as session:
        session.execute(delete(Order).where(Order.cart_id == cart_id))
        session.execute(delete(CartItem).where(CartItem.cart_id == cart_id))
        session.execute(delete(Cart).where(Cart.id == cart_id))
        session.execute(delete(Product).where(Product.product_id.in_(TEST_PRODUCTS)))
        session.commit()


def test_add_same_product_twice(cart_id):
    add_item(cart_id, "TEST-CASE")
    add_item(cart_id, "TEST-CASE", 2)
    contents = cart_contents(cart_id)
    assert contents["items"][0]["quantity"] == 3
    assert contents["total"] == 31.5


def test_cannot_add_out_of_stock(cart_id):
    with pytest.raises(ValueError, match="out of stock"):
        add_item(cart_id, "TEST-SOLD-OUT")


def test_checkout_once(cart_id):
    with pytest.raises(ValueError, match="empty"):
        checkout(cart_id)
    add_item(cart_id, "TEST-CASE")
    assert checkout(cart_id)["total"] == 10.5
    with pytest.raises(ValueError, match="already checked out"):
        add_item(cart_id, "TEST-CASE")


def test_unknown_cart(cart_id):
    with pytest.raises(ValueError, match="cart not found"):
        cart_contents(-1)
