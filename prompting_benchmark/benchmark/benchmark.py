from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import itertools
from prompting_benchmark.models import Model, HuggingfaceModel
from prompting_benchmark.benchmark import prompt_strategies, score_fns
from pathlib import Path
from typing import Optional, Union, Callable
import json
import subprocess
from tqdm import tqdm
import os
from prompting_benchmark.benchmark.task import Task


@dataclass
class BenchmarkSpec:
    model_class: str
    model_kwargs: dict
    examples_file: str
    score_fn: str
    prompt_strat: str
    prompt_strat_kwargs: dict = field(default_factory=dict)
    exemplars_prompt_strat: str = ""
    exemplars_prompt_strat_kwargs: dict = field(default_factory=dict)
    few_shot_exemplars: list[dict] = field(default_factory=list)
    max_examples: Optional[int] = None
    most_recent_commit_hash: str = ""

    def __post_init__(self):
        self.most_recent_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


class Benchmark:
    model_classes = {"HuggingfaceModel": HuggingfaceModel}
    prompt_strats = {"from_template": prompt_strategies.from_template}
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
        model = cls.model_classes[spec.model_class](**spec.model_kwargs)

        prompt_strategy = cls.prompt_strats[spec.prompt_strat](**spec.prompt_strat_kwargs)

        exemplars_prompt_strat_name = spec.exemplars_prompt_strat or spec.prompt_strat
        exemplars_prompt_strat_kwargs = spec.exemplars_prompt_strat_kwargs or spec.prompt_strat_kwargs
        exemplars_prompt_strat = cls.prompt_strats[exemplars_prompt_strat_name](**exemplars_prompt_strat_kwargs)

        task = Task.from_json(spec.examples_file, prompt_strategy, spec.few_shot_exemplars, exemplars_prompt_strat)

        score_fn = cls.score_fns[spec.score_fn]

        return cls(model, task, score_fn, spec.max_examples, spec)

    def iter_results(self, batch_size=32):
        total_examples = len(self.task) if self.max_examples is None else self.max_examples
        task_iter = iter(self.task)

        with tqdm(total=total_examples) as pbar:
            while True:
                batch = itertools.islice(task_iter, batch_size)
                prompts, targets = zip(*batch)

                for prompt, target, full_completion in zip(prompts, targets, self.model.complete(list(prompts))):
                    answer = full_completion[len(prompt) :]
                    score = self.score_fn(answer, target)
                    yield {"prompt": prompt, "target": target, "answer": answer, "score": score}
                    pbar.update()
                    if pbar.n >= total_examples:
                        return

    def write_results(self, output_file: Union[Path, str] = ""):
        if not output_file:
            timestamp = (datetime.utcnow() - timedelta(hours=7)).strftime("%Y-%m-%d--%H-%M")
            if self.spec:
                task_name = Path(self.spec.examples_file).stem
                filename = f"{self.spec.model_class}-{task_name}_{timestamp}.json"
            else:
                filename = f"benchmark_{timestamp}.json"
            output_file = Path("results") / filename

        results = {"benchmark_spec": asdict(self.spec)}
        results["results"] = list(self.iter_results())
        results["avg_score"] = sum(r["score"] for r in results["results"]) / len(results["results"])

        Path(output_file).parents[0].mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        return results


if __name__ == "__main__":
    few_shot_exemplars = [
        {
            "input": "I have a clarinet, a violin, and a flute. How many musical instruments do I have?",
            "scratchpad": "",
            "answer": "I have three musical instruments.",
        }
    ]

    spec = BenchmarkSpec(
        model_class="HuggingfaceModel",
        model_kwargs={"model_name": "EleutherAI/gpt-j-6B", "stop_tokens": ["\n", "."]},
        examples_file="tasks/object-counting.json",
        few_shot_exemplars=few_shot_exemplars,
        prompt_strat="from_template",
        prompt_strat_kwargs={"question_template": "%s\n", "answer_template": "I have%s"},
        exemplars_prompt_strat_kwargs={"question_template": "%s\n", "answer_template": "%s"},
        score_fn="target_in_answer",
    )

    benchmark = Benchmark.from_spec(spec)
    results = benchmark.write_results()
    print(results["avg_score"])
