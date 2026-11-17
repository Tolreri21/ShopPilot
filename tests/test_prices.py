import pytest

from shoppilot.data.prices import parse_price


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$29.99", 29.99),
        ("$1,299.00", 1299.0),
        ("$9.99 - $19.99", 9.99),
        ("$15", 15.0),
    ],
)
def test_parse_dollar_prices(text, expected):
    assert parse_price(text) == expected


@pytest.mark.parametrize("text", ["", "25,63€", "Currently unavailable"])
def test_not_a_dollar_price_is_none(text):
    assert parse_price(text) is None
