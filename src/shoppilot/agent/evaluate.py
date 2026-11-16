import json
import re
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from shoppilot.agent.graph import build_agent, load_llm, run
from shoppilot.agent.tools import make_tools
from shoppilot.search.service import ARTIFACTS, Searcher
from shoppilot.shop.cart import create_cart

SCENARIOS_FILE = Path(__file__).with_name("scenarios.json")
PRODUCT_ID = re.compile(r"\bB0[0-9A-Z]{8}\b")
PAUSE_SECONDS = 10


def tool_calls(messages: list) -> list[dict]:
    return [call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]


def tool_output(messages: list) -> str:
    return "\n".join(m.text for m in messages if isinstance(m, ToolMessage))


def score(case: dict, messages: list) -> dict:
    calls = tool_calls(messages)
    if case["tool"] is None:
        right_tool = not calls
    else:
        right_tool = bool(calls) and calls[0]["name"] == case["tool"]
    right_args = right_tool and all(
        calls[0]["args"].get(key) == value for key, value in case["args"].items()
    )
    mentioned = set(PRODUCT_ID.findall(messages[-1].text))
    known = set(PRODUCT_ID.findall(messages[0].text + "\n" + tool_output(messages)))
    return {
        "tool": right_tool,
        "args": right_args,
        "grounded": mentioned <= known,
        "calls": len(calls),
    }


def summary(results: list[dict]) -> dict:
    with_tool = [r for r in results if r["expected"] is not None]
    return {
        "tool accuracy": sum(r["tool"] for r in results) / len(results),
        "argument accuracy": sum(r["args"] for r in with_tool) / len(with_tool),
        "grounded answers": sum(r["grounded"] for r in results) / len(results),
        "tool calls per turn": sum(r["calls"] for r in results) / len(results),
    }


def main() -> None:
    cases = json.loads(SCENARIOS_FILE.read_text())
    searcher = Searcher()
    llm = load_llm()
    results = []
    for case in cases:
        cart_id = create_cart()
        agent = build_agent(llm, make_tools(searcher, cart_id))
        messages = run(agent, cart_id, case["message"])
        result = score(case, messages)
        result["message"] = case["message"]
        result["expected"] = case["tool"]
        result["answer"] = messages[-1].text
        results.append(result)
        print(f"{result['tool']:d} {result['args']:d} {result['grounded']:d} {case['message']}")
        time.sleep(PAUSE_SECONDS)

    print("\n| Metric | Value |\n|---|---|")
    for name, value in summary(results).items():
        print(f"| {name} | {value:.2f} |")
    (ARTIFACTS / "agent_eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
