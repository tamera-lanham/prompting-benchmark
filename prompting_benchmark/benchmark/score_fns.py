def starts_with(answer: str, targets: list[str]) -> float:
    return 1.0 if any(answer.startswith(target) for target in targets) else 0.0


def target_in_answer(answer: str, targets: list[str]) -> float:
    return 1.0 if any(target in answer for target in targets) else 0.0
