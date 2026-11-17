import json
from pathlib import Path

import numpy as np

from shoppilot.rag.faq import faq_texts, load_items
from shoppilot.search.dense import INSTRUCTION, encode, load_model

EVAL_FILE = Path(__file__).with_name("faq_eval.json")


def hit_rate(similarity: np.ndarray, targets: list[int], k: int) -> float:
    top = np.argsort(-similarity, axis=1)[:, :k]
    return float(np.mean([target in row for target, row in zip(targets, top)]))


def main() -> None:
    items = load_items()
    cases = json.loads(EVAL_FILE.read_text())
    questions = [case["question"] for case in cases]
    targets = [case["faq"] for case in cases]
    model = load_model()

    print("| FAQ text | Query instruction | Hit@1 | Hit@3 |")
    print("|---|---|---|---|")
    for name, texts in [
        ("question", [i["question"] for i in items]),
        ("question + answer", faq_texts(items)),
    ]:
        faq_vectors = encode(model, texts)
        for prefix in ["", INSTRUCTION]:
            similarity = encode(model, [prefix + q for q in questions]) @ faq_vectors.T
            hit1 = hit_rate(similarity, targets, 1)
            hit3 = hit_rate(similarity, targets, 3)
            print(f"| {name} | {'yes' if prefix else 'no'} | {hit1:.2f} | {hit3:.2f} |")


if __name__ == "__main__":
    main()
