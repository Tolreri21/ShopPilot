from pydantic import BaseModel, Field


class ProductOut(BaseModel):
    product_id: str
    title: str
    brand: str | None
    bullets: str | None
    price: float
    in_stock: bool
    score: float | None = None


class CartItemOut(BaseModel):
    product_id: str
    title: str
    price: float
    quantity: int


class CartOut(BaseModel):
    cart_id: int
    items: list[CartItemOut]
    total: float


class OrderOut(BaseModel):
    order_id: int
    total: float


class ChatIn(BaseModel):
    cart_id: int
    message: str = Field(min_length=1, max_length=2000)


class ChatOut(BaseModel):
    answer: str
