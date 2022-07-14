from typing import Callable
from word2number.w2n import word_to_num


def starts_with(answer: str, targets: list[str]) -> float:
    return 1.0 if any(answer.strip().startswith(target) for target in targets) else 0.0


def target_in_answer(answer: str, targets: list[str]) -> float:
    return 1.0 if any(target in answer for target in targets) else 0.0


def target_in_answer_line_without_char(char: str = "*") -> Callable:
    def score_fn(answer: str, targets: list[str]) -> float:
        usable_lines = "\n".join([line for line in answer.split("\n") if char not in line])
        return 1.0 if any(target in usable_lines for target in targets) else 0.0

    return score_fn


def target_in_answer_line_with_char(char: str = ">") -> Callable:
    def score_fn(answer: str, targets: list[str]) -> float:
        usable_lines = "\n".join([line for line in answer.split("\n") if char in line])
        return 1.0 if any(target in usable_lines for target in targets) else 0.0

    return score_fn
    
def target_is_first_number(answer: str, targets: list[str]) -> float:
    def is_number(string: str) -> bool:
        try:
            word_to_num(str)
        except:
            return False
        return True

    for word in answer.split(" "):
        if is_number(word):
            if any(word.strip() == target for target in targets):
                return 1.0
            return 0.0
    return 0.0

def target_is_first_number_after_char(char: str = ">") -> Callable:
    def score_fn(answer: str, targets: list[str]):
        if char in answer:
            answer = answer[answer.index(char):]
        return target_is_first_number(answer, targets)
    return score_fn