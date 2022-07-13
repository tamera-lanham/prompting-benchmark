from prompting_benchmark.benchmark.prompt_strategies import *


def test_from_template():
    ps = from_template("❓: %s\n", "🅰️: %s", "🤔 %s\n")
    prompt = ps("What is the answer?", ["Reasoning step 1", "Reasoning step 2", "Reasoning step 3"], "Answer ")
    expectation = "❓: What is the answer?\n🤔 Reasoning step 1\n🤔 Reasoning step 2\n🤔 Reasoning step 3\n🅰️: Answer "
    assert prompt == expectation
    print(prompt)
