import gzip
import json
import re

import pandas as pd

from shoppilot.config import settings


def parse_price(text: str) -> float | None:
    match = re.search(r"\$(\d[\d,]*\.?\d*)", text)
    if match is None:
        return None
    return float(match.group(1).replace(",", ""))


def load_known_prices() -> pd.DataFrame:
    rows = []
    with gzip.open(settings.data_dir / "raw/sample.json.gz", "rt") as f:
        for line in f:
            item = json.loads(line)
            if item.get("locale") != "us" or not item.get("title"):
                continue
            price = parse_price(item.get("price") or "")
            if price:
                rows.append({"title": item["title"], "price": price})
    return pd.DataFrame(rows).drop_duplicates("title").reset_index(drop=True)
