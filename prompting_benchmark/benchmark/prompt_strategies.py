# Prompting strategies
# All follow the interface:


from typing import Callable


def vanilla(question: str, scratchpad: list[str] = [], answer: str = "") -> str:
    if scratchpad:
        scratchpad_str = "\n".join(f"* {step}" for step in scratchpad)
        return f"{question}\n{scratchpad_str}\n{answer}".strip()
    return f"{question} {answer}".strip()


def q_and_a(question: str, scratchpad: list[str] = [], answer: str = "") -> str:
    q, a = f"Q: {question}", f"A: {answer}"

    if scratchpad:
        scratchpad_str = "\n".join(f"* {step}" for step in scratchpad)
        return f"{q}\n{scratchpad_str}\n{a}".strip()

    return f"{q}\n{a}".strip()


def from_template(question_template: str, answer_template: str, scratchpad_step_template: str = "* %s\n") -> Callable:
    def prompt_strategy(question: str, scratchpad: list[str] = [], answer: str = "") -> str:
        q_templated = question_template % question
        a_templated = answer_template % answer

        if scratchpad:
            scratchpad_str = "".join(scratchpad_step_template % step for step in scratchpad)
            return (q_templated + scratchpad_str + a_templated).strip()

        return (q_templated + a_templated).strip()

    return prompt_strategy
