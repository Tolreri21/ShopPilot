import re

import numpy as np
import pandas as pd

from shoppilot.config import settings

PRODUCT_GROUPS = {
    "premium": {
        "price_range": (150, 1500),
        "keywords": [
            "laptop",
            "macbook",
            "ipad",
            "tablet",
            "smartphone",
            "smart tv",
            "4k tv",
            "television",
            "projector",
            "monitor",
            "camera",
            "drone",
            "smartwatch",
            "soundbar",
            "home theater",
            "treadmill",
            "elliptical",
            "exercise bike",
            "bicycle",
            "e-bike",
            "electric scooter",
            "sofa",
            "sectional",
            "mattress",
            "recliner",
            "standing desk",
            "refrigerator",
            "dishwasher",
            "washing machine",
            "grill",
            "generator",
            "pressure washer",
            "robot vacuum",
            "air purifier",
            "espresso machine",
            "guitar",
            "digital piano",
            "telescope",
            "kayak",
            "diamond",
        ],
    },
    "mid": {
        "price_range": (25, 200),
        "keywords": [
            "sneaker",
            "running shoes",
            "boots",
            "jacket",
            "coat",
            "hoodie",
            "jeans",
            "dress",
            "handbag",
            "wallet",
            "backpack",
            "luggage",
            "sunglasses",
            "watch",
            "perfume",
            "headphones",
            "earbuds",
            "speaker",
            "keyboard",
            "webcam",
            "router",
            "printer",
            "blender",
            "air fryer",
            "coffee maker",
            "vacuum",
            "lamp",
            "mirror",
            "bookshelf",
            "office chair",
            "gaming chair",
            "tent",
            "sleeping bag",
            "dumbbell",
            "yoga mat",
            "lego",
        ],
    },
    "cheap": {
        "price_range": (2, 30),
        "keywords": [
            "case",
            "cover",
            "screen protector",
            "cable",
            "charger",
            "t-shirt",
            "tee",
            "shirt",
            "tank top",
            "leggings",
            "socks",
            "underwear",
            "hat",
            "cap",
            "gloves",
            "scarf",
            "headband",
            "keychain",
            "lanyard",
            "sticker",
            "poster",
            "pen",
            "pencil",
            "journal",
            "book",
            "mug",
            "cup",
            "bottle",
            "candle",
            "balloon",
            "costume",
            "party",
            "decoration",
            "ornament",
            "earrings",
            "bracelet",
            "necklace",
            "toy",
            "plush",
            "puzzle",
            "pillowcase",
            "towel",
            "napkin",
            "phone holder",
            "light bulb",
            "battery",
            "tape",
            "shampoo",
            "lotion",
            "soap",
            "mask",
            "vitamin",
            "supplement",
        ],
    },
    "default": {
        "price_range": (5, 100),
        "keywords": [],
    },
}


def assign_tier(title: str) -> str:
    lower_title = title.lower()
    for group in ["cheap", "mid", "premium"]:
        for word in PRODUCT_GROUPS[group]["keywords"]:
            if re.search(rf"\b{word}s?\b", lower_title):
                return group
    return "default"


def main():
    rng = np.random.default_rng(settings.seed)
    products = pd.read_parquet(
        settings.data_dir / "processed/products.parquet",
    )
    products["tier"] = products["product_title"].apply(assign_tier)

    low = products["tier"].map(lambda t: PRODUCT_GROUPS[t]["price_range"][0])
    high = products["tier"].map(lambda t: PRODUCT_GROUPS[t]["price_range"][1])

    products["price"] = rng.uniform(low, high)
    products["price"] = products["price"].round(2)
    products["price_source"] = "rules"
    products["in_stock"] = rng.random(products.shape[0]) > 0.12

    products[["product_id", "price", "price_source", "in_stock"]].to_parquet(
        settings.data_dir / "processed/store.parquet"
    )
    print(products["tier"].value_counts())
    print(products.groupby("tier")["price"].median())


if __name__ == "__main__":
    main()
