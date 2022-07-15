from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import time
import torch as t

from prompting_benchmark.benchmark.benchmark import Benchmark
from prompting_benchmark.benchmark.prompt_strategies import (
    from_template,
    from_template_no_scratchpad,
    from_template_replace_scratchpad,
)
from prompting_benchmark.benchmark.score_fns import (
    target_in_answer,
    target_in_answer_line_with_char,
    target_is_first_number,
    target_is_first_number_after_char,
)
from prompting_benchmark.benchmark.task import Task
from prompting_benchmark.models.huggingface import HuggingfaceModel


prompt_strat_options = {
    "no_scratchpad": {
        "prompt_strategy": from_template_no_scratchpad("Q: %s\n", "A: I have%s"),
        "exemplars_prompt_strategy": from_template_no_scratchpad("Q: %s\n", "A: %s"),
    },
    "scratchpad": {
        "prompt_strategy": from_template("%s", "\n*%s"),
        "exemplars_prompt_strategy": from_template("%s", "\n> %s", "\n* %s"),
    },
    "ellipsis": {
        "prompt_strategy": from_template_replace_scratchpad(
            "Q: %s\n", "\n".join("...".join(" " for word in range(20)) for step in range(5)) + "\nA: I have%s"
        ),
        "exemplars_prompt_strategy": from_template_replace_scratchpad("Q: %s\n", "\nA: %s", "..."),
    },
}

score_fn_options = {
    "target_is_first_number": target_is_first_number,
    "target_is_first_number_after_char_>": target_is_first_number_after_char(">"),
}


def get_task(task_def, few_shot_exemplars):
    exemplars = few_shot_exemplars[: task_def["n_exemplars"]]
    score_fn = score_fn_options[task_def["score_fn_option"]]
    return (
        Task.from_json(
            task_def["task_file"], few_shot_exemplars=exemplars, **prompt_strat_options[task_def["prompt_strat_option"]]
        ),
        score_fn,
    )


timestamp = (datetime.utcnow() - timedelta(hours=7)).strftime("%Y-%m-%d--%H-%M")
output_dir = Path("results") / ("benchmark-suite_" + timestamp)

with open("tasks/benchmark-suite-defs/object-counting.json") as f:
    benchmark_suite_defs = json.load(f)

model_name_options = benchmark_suite_defs["model_name_options"]
few_shot_exemplars = benchmark_suite_defs["few_shot_exemplars"]
task_options = benchmark_suite_defs["tasks"]

start = time()
scores = {}
for model_name in model_name_options:
    model = HuggingfaceModel(model_name, device="cuda:0")

    for task_def in task_options:

        # if task_def["task_id"] not in [4, 5, 6]:
        #     continue

        task, score_fn = get_task(task_def, few_shot_exemplars)
        benchmark = Benchmark(model, task, score_fn)

        model.set_stop_tokens(task_def["stop_tokens"])

        filename = output_dir / (
            model_name.replace(".", "-").split("/")[-1] + "_task-" + str(task_def["task_id"]) + ".json"
        )
        print(filename.stem)
        results = benchmark.write_results(filename, additional_info=[model_name, task_def])
        scores[filename.stem] = results["avg_score"]

    del model
    t.cuda.empty_cache()

end = time()
total_mins = (end - start) / 60

print(total_mins)
print(scores)

with open(output_dir / "results.json", "w") as f:
    json.dump({"scores": scores, "total_mins": total_mins}, f)
