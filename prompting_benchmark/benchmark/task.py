from pathlib import Path
from typing import Callable, Union, Iterable, Tuple
from prompting_benchmark.benchmark.prompt_strategies import q_and_a, vanilla
import json


class Task:
    def __init__(
        self,
        examples: Iterable[dict],  # dict: {"input": str, "target": list}
        prompt_strategy: Callable = vanilla,
        few_shot_exemplars: Iterable[dict] = [],  # dict: {"input": str, "scratchpad": list, "answer": str}
    ):
        self.prompt_strategy = prompt_strategy
        self.examples = examples
        self.few_shot_exemplars = few_shot_exemplars

    @classmethod
    def from_json(
        cls,
        task_file: Union[Path, str],
        prompt_strategy: Callable = vanilla,
        few_shot_exemplars: Iterable[dict] = [],
    ):
        with open(task_file) as f:
            task_spec = json.load(f)
        return cls(task_spec["examples"], prompt_strategy, few_shot_exemplars)

    def __iter__(self):
        few_shot_prompt = "".join(
            self.prompt_strategy(ex["input"], ex.get("scratchpad", []), ex.get("answer", "")) + "\n\n"
            for ex in self.few_shot_exemplars
        )
        for ex in self.examples:
            yield few_shot_prompt + self.prompt_strategy(ex["input"]), ex["target"]

    def __len__(self):
        return len(self.examples)


if __name__ == "__main__":
    task = Task.from_json(Path("tasks") / "object-counting.json")
    t = list(task)
    print(t[0])
