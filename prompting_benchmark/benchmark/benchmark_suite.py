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

few_shot_exemplars = [
    {
        "input": "I have an apple, three oranges, and a banana. How many fruits do I have?",
        "scratchpad": [
            "An apple is a fruit, and I have one. I have one fruit.",
            "An orange is a fruit, and I have three. I have one fruit plus three, so I have four fruits.",
            "A banana is a fruit, and I have one. I have four fruits plus one, so I have five fruits.",
        ],
        "answer": "I have five fruits.",
    },
    {
        "input": "I have an accordion, a yam, two clarinets, a garlic, a cauliflower, a cabbage, and a flute. How many vegetables do I have?",
        "scratchpad": [
            "An accordion is not a vegetable, so I will ignore it.",
            "A yam is a vegetable, and I have one. I have one vegetable.",
            "A clairnet is not a vegetable, so I will ignore it.",
            "A garlic is a vegetable, and I have one. I have one vegetable plus one, so I have two vegetables.",
            "A cauliflower is a vegetable, and I have one. I have two vegetables plus one, so I have three vegetables.",
            "A cabbage is a vegetable, and I have one. I have three vegetables plus one, so I have four vegetables.",
        ],
        "answer": "I have four vegetables.",
    },
    {
        "input": "I have a flute, a mouse, and a duck. How many animals do I have?",
        "scratchpad": [
            "A flute is not an animal, so I will ignore it.",
            "A mouse is an animal, and I have one. I have one animal.",
            "A duck is an animal, and I have one. I have one animal plus one, so I have two animals.",
        ],
        "answer": "I have two animals.",
    },
    {
        "input": "I have three pianos, four clarinets, and a flute. How many musical instruments do I have?",
        "scratchpad": [
            "A piano is a musical instrument, and I have three. I have three musical instruments.",
            "A clarinet is a musical instrument, and I have four. I have four musical instruments plus one, so I have five musical instruments.",
            "A flute is a musical instrument, and I have one. I have five musical instruments plus one, so I have six musical instruments.",
        ],
        "answer": "I have eight musical instruments.",
    },
    {
        "input": "I have two dogs, a cow, and a goose. How many animals do I have?",
        "scratchpad": [
            "A dog is an animal, and I have two. I have two animals.",
            "A cow is an animal, and I have one. I have two animals plus one, so I have three animals.",
            "A goose is an animal, and I have one. I have three animals plus one, so I have four animals.",
        ],
        "answer": "I have four animals.",
    },
    {
        "input": "I have three grapes, a pomegranate, a couch, and a pear. How many fruits do I have?",
        "scratchpad": [
            "A grape is a fruit, and I have three. I have three fruits.",
            "A pomegranate is a fruit, and I have one. I have three fruits plus one, so I have four fruits.",
            "A couch is not a fruit. I will ignore it.",
            "A pear is a fruit, and I have one. I have four fruits plus one, so I have five fruits.",
        ],
        "answer": "I have five fruits.",
    },
    {
        "input": "I have a guitar, a trumpet, and a drum. How many musical instruments do I have?",
        "scratchpad": [
            "A guitar is a musical instrument, and I have one. I have one musical instrument.",
            "A trumpet is a musical instrument, and I have one. I have one musical instrument plus one, so I have two musical instruments.",
            "A drum is a musical instrument, and I have one. I have two musical instruments plus one, so I have three musical instruments.",
        ],
        "answer": "I have three musical instruments.",
    },
    {
        "input": "I have a cabbage, a pumpkin, a cauliflower, and a table. How many vegetables do I have?",
        "scratchpad": [
            "A cabbage is a vegetable, and I have one. I have one vegetable.",
            "A pumpkin is a vegetable, and I have one. I have one vegetable plus one, so I have two vegetables.",
            "A cauliflower is a vegetable, and I have one. I have two vegetables plus one, so I have three vegetables.",
            "A table is not a vegetable, so I will ignore it.",
        ],
        "answer": "I have three vegetables.",
    },
    {
        "input": "I have a cat, a dog, and a mouse. How many animals do I have?",
        "scratchpad": [
            "A cat is an animal, and I have one. I have one animal.",
            "A  dog is an animal, and I have one. I have one animal plus one, so I have two animals.",
            "A mouse is an animal, and I have one. I have two animals plus one, so I have three animals.",
        ],
        "answer": "I have three animals.",
    },
    {
        "input": "I have a lamp, two cars, a couch, and a microwave. How many objects do I have?",
        "scratchpad": [
            "A lamp is an object, and I have one. I have one object.",
            "A car is an object, and I have two. I have two objects plus one, so I have three objects.",
            "A couch is an object, and I have one. I have three objects plus one, so I have four objects.",
            "A microwave is an object, and I have one. I have four objects plus one, so I have five objects.",
        ],
        "answer": "I have five objects.",
    },
    {
        "input": "I have an accordion, a flute, a trumpet, and a guitar. How many musical instruments do I have?",
        "scratchpad": [
            "An accordion is a musical instrument, and I have one. I have one musical instrument.",
            "A flute is a musical instrument, and I have one. I have one musical instrument plus one, so I have two musical instruments.",
            "A trumpet is a musical instrument, and I have one. I have two musical instruments plus one, so I have three musical instruments.",
            "A guitar is a musical instrument, and I have one. I have three musical instruments plus one, so I have four musical instruments.",
        ],
        "answer": "I have four musical instruments.",
    },
]


prompt_strat_options = {
    "no_scratchpad": {
        "prompt_strategy": from_template_no_scratchpad("%s\n", "I have%s"),
        "exemplars_prompt_strategy": from_template_no_scratchpad("%s\n", "%s"),
    },
    "scratchpad": {
        "prompt_strategy": from_template("%s", "\n*%s"),
        "exemplars_prompt_strategy": from_template("%s", "\n> %s", "\n* %s"),
    },
    "ellipsis": {
        "prompt_strategy": from_template_replace_scratchpad(
            "%s\n", "\n".join("...".join(" " for word in range(30)) for step in range(5)) + "\n> I have%s"
        ),
        "exemplars_prompt_strategy": from_template_replace_scratchpad("%s\n", "\n> %s", "..."),
    },
}

score_fn_options = {
    "target_is_first_number": target_is_first_number,
    "target_is_first_number_after_char_>": target_is_first_number_after_char(">"),
}


def get_task(task_file, n_exemplars, prompt_strat_option, score_fn_option, **_):
    exemplars = few_shot_exemplars[:n_exemplars]
    score_fn = score_fn_options[score_fn_option]
    return (
        Task.from_json(task_file, few_shot_exemplars=exemplars, **prompt_strat_options[prompt_strat_option]),
        score_fn,
    )


task_file = "tasks/object-counting.json"

task_options = [
    {
        "task_file": task_file,
        "n_exemplars": 0,
        "prompt_strat_option": "no_scratchpad",
        "score_fn_option": "target_is_first_number",
        "stop_tokens": ["\n"],
    },
    {
        "task_file": task_file,
        "n_exemplars": 1,
        "prompt_strat_option": "no_scratchpad",
        "score_fn_option": "target_is_first_number",
        "stop_tokens": ["\n"],
    },
    {
        "task_file": task_file,
        "n_exemplars": 3,
        "prompt_strat_option": "no_scratchpad",
        "score_fn_option": "target_is_first_number",
        "stop_tokens": ["\n"],
    },
    {
        "task_file": task_file,
        "n_exemplars": 1,
        "prompt_strat_option": "scratchpad",
        "score_fn_option": "target_is_first_number_after_char_>",
        "stop_tokens": [],
    },
    {
        "task_file": task_file,
        "n_exemplars": 3,
        "prompt_strat_option": "scratchpad",
        "score_fn_option": "target_is_first_number_after_char_>",
        "stop_tokens": [],
    },
    {
        "task_file": task_file,
        "n_exemplars": 1,
        "prompt_strat_option": "ellipsis",
        "score_fn_option": "target_is_first_number",
        "stop_tokens": ["\n"],
    },
    {
        "task_file": task_file,
        "n_exemplars": 3,
        "prompt_strat_option": "ellipsis",
        "score_fn_option": "target_is_first_number",
        "stop_tokens": ["\n"],
    },
]

model_name_options = ["EleutherAI/gpt-j-6B", "EleutherAI/gpt-neo-1.3B", "gpt2"]

timestamp = (datetime.utcnow() - timedelta(hours=7)).strftime("%Y-%m-%d--%H-%M")
output_dir = Path("results") / ("benchmark-suite_" + timestamp)

start = time()
scores = {}
for model_name in model_name_options:
    model = HuggingfaceModel(model_name)

    for i, task_kwargs in enumerate(task_options):
        task, score_fn = get_task(**task_kwargs)
        benchmark = Benchmark(model, task, score_fn)

        if "stop_tokens" in task_kwargs:
            model.set_stop_tokens(task_kwargs["stop_tokens"])

        filename = output_dir / (model_name.replace(".", "-").split("/")[-1] + "_task-" + str(i) + ".json")
        print(filename.stem)
        results = benchmark.write_results(filename, additional_info=[model_name, task_kwargs])
        scores[filename.stem] = results["avg_score"]

    del model
    t.cuda.empty_cache()

end = time()
total_mins = (end - start) / 60

print(total_mins)
print(scores)

with open(output_dir / "results.json") as f:
    json.dump({"scores": scores, "total_mins": total_mins}, f)
