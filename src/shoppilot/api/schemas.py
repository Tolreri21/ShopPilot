from pydantic import BaseModel


class ProductOut(BaseModel):
    product_id: str
    title: str
    brand: str | None
    bullets: str | None
    price: float
    in_stock: bool
    score: float | None = None
