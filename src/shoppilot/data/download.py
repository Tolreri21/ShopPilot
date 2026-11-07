from pathlib import Path

import httpx
from tqdm import tqdm

from shoppilot.config import settings

ESCI = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"

URLS = [
    f"{ESCI}/shopping_queries_dataset_examples.parquet",
    f"{ESCI}/shopping_queries_dataset_products.parquet",
    f"{ESCI}/shopping_queries_dataset_sources.csv",
    "https://github.com/shuttie/esci-s/raw/master/sample.json.gz",
]


def download(url: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_bytes(1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"{path.name} is incomplete, run again")
    tmp.rename(path)


def main() -> None:
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        name = url.split("/")[-1]
        path = raw_dir / name
        if path.exists():
            print(f"skip {name}, already downloaded")
            continue
        print(f"downloading {name}")
        download(url, path)


if __name__ == "__main__":
    main()
