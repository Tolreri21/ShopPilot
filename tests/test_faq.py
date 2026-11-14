import json

import numpy as np

from shoppilot.rag.faq import faq_texts, load_items
from shoppilot.rag.faq_eval import EVAL_FILE, hit_rate


def test_faq_items_have_question_and_answer():
    items = load_items()
    assert len(items) == len({item["question"] for item in items})
    assert all(item["question"] and item["answer"] for item in items)


def test_faq_text_joins_question_and_answer():
    assert faq_texts([{"question": "Q?", "answer": "A."}]) == ["Q? A."]


def test_hit_rate():
    similarity = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.8]])
    assert hit_rate(similarity, [0, 1], 1) == 0.5
    assert hit_rate(similarity, [0, 1], 2) == 1.0


def test_eval_targets_exist():
    cases = json.loads(EVAL_FILE.read_text())
    assert all(0 <= case["faq"] < len(load_items()) for case in cases)
