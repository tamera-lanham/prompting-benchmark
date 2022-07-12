from dataclasses import asdict, dataclass
from datetime import datetime
from prompting_benchmark.models import Model
from prompting_benchmark import models
from prompting_benchmark.benchmark import prompt_strategies, score_fns
from pathlib import Path
from typing import Optional, Union, Callable
import json
from tqdm import tqdm
import os
from prompting_benchmark.benchmark.task import Task


@dataclass
class BenchmarkSpec:
    model: str
    model_kwargs: dict
    examples_file: str
    few_shot_exemplars: list[dict]
    prompt_templates: dict
    score_fn: str
    max_examples: Optional[int] = None


class Benchmark:
    models = {"GPT2": models.GPT2, "GPTJ": models.GPTJ}
    score_fns = {"starts_with": score_fns.starts_with, "target_in_answer": score_fns.target_in_answer}

    def __init__(
        self,
        model: Model,
        task: Task,
        score_fn: Callable,
        max_examples: Optional[int] = None,
        spec: Optional[BenchmarkSpec] = None,
    ):
        self.model = model
        self.task = task
        self.score_fn = score_fn
        self.max_examples = max_examples
        self.spec = spec

    @classmethod
    def from_spec(cls, spec: BenchmarkSpec):
        model = cls.models[spec.model](**spec.model_kwargs)
        prompt_strategy = prompt_strategies.from_template(**spec.prompt_templates)
        task = Task.from_json(spec.examples_file, prompt_strategy, spec.few_shot_exemplars)
        score_fn = cls.score_fns[spec.score_fn]
        return cls(model, task, score_fn, spec.max_examples, spec)

    def iter_results(self):
        total_examples = len(self.task) if self.max_examples is None else self.max_examples

        for i, (prompt, target) in tqdm(enumerate(self.task), total=total_examples):
            if i >= total_examples:
                break

            full_completion = self.model.complete([prompt])[0]
            answer = full_completion[len(prompt) :]
            score = self.score_fn(answer, target)

            yield {"prompt": prompt, "target": target, "answer": answer, "score": score}

    def write_results(self, output_file: Union[Path, str] = ""):
        if not output_file and self.spec:
            timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M")
            output_file = Path("results") / f"{timestamp}.json"

        results = {"benchmark_spec": asdict(self.spec)}
        results["results"] = list(self.iter_results())
        results["avg_score"] = sum(r["score"] for r in results["results"]) / len(results["results"])

        Path(output_file).parents[0].mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        return results["avg_score"]


if __name__ == "__main__":
    spec = BenchmarkSpec(
        model="GPTJ",
        model_kwargs={"stop_tokens": ["\n"]},
        examples_file="tasks/object-counting.json",
        few_shot_exemplars=[],
        prompt_templates={"question_template": "Q: %s\n", "answer_template": "A: %s"},
        score_fn="target_in_answer",
        max_examples=100,
    )

    benchmark = Benchmark.from_spec(spec)
    score = benchmark.write_results()
    print(score)
