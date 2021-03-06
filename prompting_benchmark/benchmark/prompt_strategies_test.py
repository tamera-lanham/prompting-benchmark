from prompting_benchmark.benchmark.prompt_strategies import *


def test_from_template():
    ps = from_template("ā: %s\n", "š°ļø: %s", "š¤ %s\n")
    prompt = ps("What is the answer?", ["Reasoning step 1", "Reasoning step 2", "Reasoning step 3"], "Answer ")
    expectation = "ā: What is the answer?\nš¤ Reasoning step 1\nš¤ Reasoning step 2\nš¤ Reasoning step 3\nš°ļø: Answer "
    assert prompt == expectation
    print(prompt)
