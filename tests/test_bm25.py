import pandas as pd

from shoppilot.search.bm25 import bm25_scores, build_index, product_texts


def test_product_texts_handles_missing_values():
    products = pd.DataFrame(
        {
            "product_title": ["Phone case"],
            "product_brand": [None],
            "product_bullet_point": [None],
        }
    )
    assert product_texts(products) == ["Phone case  "]


def test_matching_document_gets_highest_score():
    retriever = build_index(["red running shoes", "blue phone case", "running socks"])
    scores = bm25_scores(retriever, "running shoes")
    assert scores.argmax() == 0


def test_unknown_words_give_zero():
    retriever = build_index(["red running shoes", "blue phone case"])
    assert bm25_scores(retriever, "laptop").sum() == 0
