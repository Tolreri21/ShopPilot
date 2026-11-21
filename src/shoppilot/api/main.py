from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from shoppilot.agent.graph import ask, build_agent, load_llm
from shoppilot.agent.tools import make_tools
from shoppilot.api.schemas import CartOut, ChatIn, ChatOut, OrderOut, ProductOut
from shoppilot.search.service import Searcher
from shoppilot.shop import cart

app = FastAPI(title="ShopPilot")
INDEX_PAGE = Path(__file__).with_name("static") / "index.html"


@lru_cache
def get_searcher() -> Searcher:
    return Searcher()


@lru_cache
def get_llm():
    try:
        return load_llm()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))


def shop_error(error: ValueError) -> HTTPException:
    status = 404 if "not found" in str(error) else 400
    return HTTPException(status_code=status, detail=str(error))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_PAGE)


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


@app.post("/carts")
def create_cart() -> dict:
    return {"cart_id": cart.create_cart()}


@app.get("/carts/{cart_id}", response_model=CartOut)
def get_cart(cart_id: int):
    try:
        return cart.cart_contents(cart_id)
    except ValueError as error:
        raise shop_error(error)


@app.delete("/carts/{cart_id}/items/{product_id}", response_model=CartOut)
def remove_from_cart(cart_id: int, product_id: str):
    try:
        cart.remove_item(cart_id, product_id)
        return cart.cart_contents(cart_id)
    except ValueError as error:
        raise shop_error(error)


@app.post("/carts/{cart_id}/checkout", response_model=OrderOut)
def checkout(cart_id: int):
    try:
        return cart.checkout(cart_id)
    except ValueError as error:
        raise shop_error(error)


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, searcher: Searcher = Depends(get_searcher), llm=Depends(get_llm)):
    try:
        cart.cart_contents(body.cart_id)
    except ValueError as error:
        raise shop_error(error)
    agent = build_agent(llm, make_tools(searcher, body.cart_id))
    return {"answer": ask(agent, body.cart_id, body.message)}
