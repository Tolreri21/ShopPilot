from shoppilot.rag.faq import faq_texts, load_items


def test_faq_items_have_question_and_answer():
    items = load_items()
    assert len(items) == len({item["question"] for item in items})
    assert all(item["question"] and item["answer"] for item in items)


def test_faq_text_joins_question_and_answer():
    assert faq_texts([{"question": "Q?", "answer": "A."}]) == ["Q? A."]
