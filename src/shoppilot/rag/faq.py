import json
from pathlib import Path

from sqlalchemy import delete, select

from shoppilot.db.models import Faq
from shoppilot.db.session import Session
from shoppilot.search.dense import encode, load_model

FAQ_FILE = Path(__file__).with_name("faq.json")


def load_items() -> list[dict]:
    return json.loads(FAQ_FILE.read_text())


def faq_texts(items: list[dict]) -> list[str]:
    return [item["question"] + " " + item["answer"] for item in items]


def nearest_faq(vector, k: int = 3) -> list[dict]:
    distance = Faq.embedding.cosine_distance(vector)
    with Session() as session:
        rows = session.scalars(select(Faq).order_by(distance).limit(k)).all()
        return [{"question": row.question, "answer": row.answer} for row in rows]


def main() -> None:
    items = load_items()
    vectors = encode(load_model(), faq_texts(items))
    with Session() as session:
        session.execute(delete(Faq))
        for item, vector in zip(items, vectors):
            session.add(Faq(question=item["question"], answer=item["answer"], embedding=vector))
        session.commit()
    print(f"loaded {len(items)} faq entries")


if __name__ == "__main__":
    main()
