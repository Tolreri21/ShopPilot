import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shoppilot.agent.evaluate import SCENARIOS_FILE, score, summary


def search_turn(answer: str) -> list:
    return [
        HumanMessage("knife set under $40"),
        AIMessage(
            "",
            tool_calls=[{"name": "search_products", "args": {"max_price": 40}, "id": "1"}],
        ),
        ToolMessage("B005JVK1PE | Knife set | $28.50 | in stock", tool_call_id="1"),
        AIMessage(answer),
    ]


CASE = {"message": "knife set under $40", "tool": "search_products", "args": {"max_price": 40}}


def test_right_tool_and_args():
    result = score(CASE, search_turn("Try B005JVK1PE for $28.50."))
    assert result == {"tool": True, "args": True, "grounded": True, "calls": 1}


def test_wrong_args():
    result = score(CASE | {"args": {"max_price": 30}}, search_turn("ok"))
    assert result["tool"] and not result["args"]


def test_made_up_product_is_not_grounded():
    assert not score(CASE, search_turn("Try B0FAKE1234."))["grounded"]


def test_no_tool_expected():
    case = {"message": "hi", "tool": None, "args": {}}
    assert score(case, [HumanMessage("hi"), AIMessage("Hello!")])["tool"]
    assert not score(case, search_turn("Hello!"))["tool"]


def test_summary_counts_args_only_for_tool_cases():
    results = [
        {"expected": "shop_faq", "tool": True, "args": True, "grounded": True, "calls": 1},
        {"expected": None, "tool": False, "args": False, "grounded": True, "calls": 2},
    ]
    assert summary(results) == {
        "tool accuracy": 0.5,
        "argument accuracy": 1.0,
        "grounded answers": 1.0,
        "tool calls per turn": 1.5,
    }


def test_scenarios_use_known_tools():
    tools = {"search_products", "product_details", "add_to_cart", "view_cart", "shop_faq", None}
    assert all(case["tool"] in tools for case in json.loads(SCENARIOS_FILE.read_text()))
