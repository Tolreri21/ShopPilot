from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DIMENSIONS = 384


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    brand: Mapped[str | None]
    bullets: Mapped[str | None]
    price: Mapped[float]
    price_source: Mapped[str]
    in_stock: Mapped[bool]
    embedding = mapped_column(Vector(DIMENSIONS))


class Cart(Base):
    __tablename__ = "carts"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CartItem(Base):
    __tablename__ = "cart_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    cart_id: Mapped[int] = mapped_column(ForeignKey("carts.id"))
    product_id: Mapped[str] = mapped_column(ForeignKey("products.product_id"))
    quantity: Mapped[int] = mapped_column(default=1)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    cart_id: Mapped[int] = mapped_column(ForeignKey("carts.id"))
    total: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str]
    answer: Mapped[str]
    embedding = mapped_column(Vector(DIMENSIONS))
