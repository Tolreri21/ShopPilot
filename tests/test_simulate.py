from shoppilot.data.simulate import assign_tier


def test_expensive_product_is_premium():
    assert assign_tier("Samsung 55 inch smart tv") == "premium"


def test_accessory_is_cheap():
    assert assign_tier("iPhone 12 case") == "cheap"


def test_keyword_inside_another_word_is_ignored():
    assert assign_tier("Showcase display stand") != "cheap"


def test_unknown_product_is_default():
    assert assign_tier("Mystery item") == "default"
