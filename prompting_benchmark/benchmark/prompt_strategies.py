from typing import Callable


def from_template(question_template: str, answer_template: str, scratchpad_step_template: str = "* %s\n") -> Callable:
    def prompt_strategy(question: str, scratchpad: list[str] = [], answer: str = "") -> str:
        q_templated = question_template % question
        a_templated = answer_template % answer

        if scratchpad:
            scratchpad_str = "".join(scratchpad_step_template % step for step in scratchpad)
            return q_templated + scratchpad_str + a_templated

        return q_templated + a_templated

    return prompt_strategy
