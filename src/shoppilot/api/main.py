from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query

from shoppilot.api.schemas import ProductOut
from shoppilot.search.service import Searcher

app = FastAPI(title="ShopPilot")


@lru_cache
def get_searcher() -> Searcher:
    return Searcher()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search", response_model=list[ProductOut])
def search(
    q: str = Query(min_length=1),
    max_price: float | None = Query(None, gt=0),
    in_stock: bool = False,
    limit: int = Query(10, ge=1, le=50),
    searcher: Searcher = Depends(get_searcher),
):
    return searcher.search(q, max_price, in_stock, limit)


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str, searcher: Searcher = Depends(get_searcher)):
    product = searcher.get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="product not found")
    return product
